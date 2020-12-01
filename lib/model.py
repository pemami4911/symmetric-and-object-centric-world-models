import torch
import torch.nn as nn
import math
from sacred import Ingredient
from lib.layers import Flatten, GatedConv2dBN
from lib.utils import init_weights, _softplus_to_std, mvn, std_mvn
from lib.utils import gmm_loglikelihood, gaussian_loglikelihood
from lib.geco import GECO

import numpy as np

net = Ingredient('Net')

@net.config
def cfg():
    input_size = [3,64,64] # [C, H, W]
    z_size = 64
    K = 4
    time_inference_schedule = ["L"]
    iterative_inference_schedule = [4]
    log_scale = math.log(0.11)  # log base e
    bg_log_scale = math.log(0.09)
    refinenet_channels_in = 17
    refinenet_conv_size = 5
    iterative_VI_projection = True
    separate_variances = True
    lstm_dim = 128
    conv_channels = 32
    stochastic_samples = 1
    kl_beta = 1
    geco_warm_start = 1000
    action_dim = 4
    action_conditional_dynamics = False
    ssm = 'RSSM' 
    dynamics_uncertainty = False
    action_shift = 0
    action_noise = 0.05


class RelationalDynamics(nn.Module):
    @net.capture
    def __init__(self, z_size, K, action_dim, action_conditional_dynamics=False, ssm='RSSM'):
        super(RelationalDynamics, self).__init__()
        self.z_size = z_size
        self.K = K
        self.action_conditional=action_conditional_dynamics
        self.dim = 2 * z_size
        self.a_dim = action_dim
        self.ssm = ssm

        self.d_obj = nn.Sequential(
                nn.Linear(self.z_size, self.dim),
                nn.ELU(True)
            )
        self.d_obj_eff = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim * 2),
                nn.ELU(True)
            )
        self.d_obj_att = nn.Sequential(
                nn.Linear(self.dim * 2, 1),
                nn.Sigmoid()
            )
        
        self.d_comb = nn.Linear(self.dim + (2*self.dim), self.z_size)
        self.loc = nn.Linear(self.z_size, self.z_size)
        self.scale = nn.Linear(self.z_size, self.z_size)
        if self.ssm == 'Ours':
            self.past_rnn = nn.GRU(self.dim, self.dim)
        elif self.ssm == 'RSSM':
            self.past_rnn = nn.GRU(self.dim, self.z_size)
            self.h = nn.Linear(self.z_size, self.z_size)

        if self.action_conditional:
            self.d_act = nn.Sequential(
                    nn.Linear(self.a_dim, 32),
                    nn.ELU(True)
                )
            self.d_act_eff = nn.Sequential(
                    nn.Linear(self.dim + 32, self.dim),
                    nn.ELU(True)
                )
            self.d_act_att = nn.Sequential(
                    nn.Linear(self.dim, 1),
                    nn.Sigmoid()
                )


    def interaction_attention(self, z, a):
        # object embedding
        z_embed = self.d_obj(z)
        if self.ssm == 'Ours':
            self.past_rnn.flatten_parameters()
            z_embed, (h,c) = self.past_rnn(z_embed)
        z_embed = z_embed[-1]

        z_embed = z_embed.view(-1, self.K, self.dim)
        #z = z.view(-1, self.K, self.z_size)
        a_embed = self.d_act(a)
        # repeat over K and concat to z
        a_embed = a_embed.unsqueeze(1).repeat(1, self.K, 1)  # [batch_size, K, 32]
        za = torch.cat([z_embed, a_embed], 2)  # [batch_size, K, self.dim + 32]
        za_eff = self.d_act_eff(za)  # [batch_size, K, self.dim]
        za_att = self.d_act_att(za_eff)  # [batch_size, K, 1]
        z_embed = za_att * za_eff  # [batch_size, K, self.dim]

        # TODO: make more efficient for j in range(i,self.K) etc
        interaction_attention = []
        for i in range(self.K):
            z_eff = []
            z_att = []
            for j in range(self.K):
                if i == j:
                    continue
                # compute the effect and attention
                zed = torch.cat([z_embed[:,i], z_embed[:,j]], 1)
                eff = self.d_obj_eff(zed)
                att = self.d_obj_att(eff)
                z_eff += [eff]
                z_att += [att]
            # stack along K
            # shape = [batch_size, K-1, 256]
            z_eff = torch.stack(z_eff, 1)
            # shape = [batch_size, K-1, 1]
            z_att = torch.stack(z_att, 1)
            interaction_attention += [z_att]
        # shape = [batch_size * K, K-1, 1]
        interaction_attention = torch.cat(interaction_attention)
        return interaction_attention


    def forward(self, z, h, a=None):
        """
        1. embed each of the K objects
        2. compute effects for all i \neq j in K
        3. compute attention for effects
        4. take inner product of effects with attention to produce 256-dim
            relation vectors for each zm
        5. concat with z and project to zm_size

        Args:
            z is [batch_size * K, z_size]
            a is [batch_size, a_dim]
        """

        relations = []

        # object embedding
        z_embed = self.d_obj(z)

        # temporal step from t-1 to t
        if self.ssm == 'RSSM':
            self.past_rnn.flatten_parameters()
            z_embed, h = self.past_rnn(z_embed.unsqueeze(0), h)
            z_embed = z_embed[-1]
            z_embed = z_embed.view(-1, self.K, self.z_size)
            h = h.view(-1, self.K, self.z_size)
            z_embed = torch.cat([z_embed, h], 2) # [batch_size, K, self.dim]
        elif self.ssm == 'Ours':
            self.past_rnn.flatten_parameters()
            z_embed, h = self.past_rnn(z_embed)
            h = h[-1]
            z_embed = z_embed[-1]
            z_embed = z_embed.view(-1, self.K, self.dim)
        elif self.ssm == 'SSM':
            z_embed = z_embed.view(-1, self.K, self.dim)
            h = h[-1]

        # add the action embedding
        if self.action_conditional:
            a_embed = self.d_act(a)
            # repeat over K and concat to z
            a_embed = a_embed.unsqueeze(1).repeat(1, self.K, 1)  # [batch_size, K, 32]
            za = torch.cat([z_embed, a_embed], 2)  # [batch_size, K, self.dim + 32]
            za_eff = self.d_act_eff(za)  # [batch_size, K, self.dim]
            za_att = self.d_act_att(za_eff)  # [batch_size, K, 1]
            z_embed = za_att * za_eff  # [batch_size, K, self.dim]
        
        eff_cache = {}
        att_cache = {}
        for i in range(self.K):
            for j in range(i+1,self.K):
                # compute the effect and attention
                zed = torch.cat([z_embed[:,i], z_embed[:,j]], 1)
                eff = self.d_obj_eff(zed)
                att = self.d_obj_att(eff)
                eff_cache['{}->{}'.format(i,j)] = eff
                att_cache['{}->{}'.format(i,j)] = att
        

        for i in range(self.K):
            z_eff = []
            z_att = []
            for j in range(self.K):
                if j > i:
                    z_eff += [eff_cache['{}->{}'.format(i,j)]]
                    z_att += [att_cache['{}->{}'.format(i,j)]]
                elif i > j:
                    z_eff += [eff_cache['{}->{}'.format(j,i)]]
                    z_att += [att_cache['{}->{}'.format(j,i)]]
            # stack along K
            # shape = [batch_size, K-i+1, 256]
            z_eff = torch.stack(z_eff, 1)
            # shape = [batch_size, K-i+1, 1]
            z_att = torch.stack(z_att, 1)
            E = (z_att * z_eff).sum(1) # sum along slot dim
            # combine
            rel = self.d_comb(torch.cat([z_embed[:,i], E], 1))
            relations += [rel]
        relations = torch.stack(relations, 1).view(-1, self.z_size)
        loc = self.loc(relations)
        scale = self.scale(relations)
        if self.ssm == 'RSSM':
            # deterministic state
            h = self.h(relations)
        return torch.cat([loc, scale], 1), h


class RefinementNetwork(nn.Module):
    @net.capture
    def __init__(self, z_size, input_size, refinenet_channels_in, refinenet_conv_size, conv_channels, lstm_dim, iterative_VI_projection):
        super(RefinementNetwork, self).__init__()
        self.input_size = input_size
        self.z_size = z_size

        self.projection = iterative_VI_projection
        if self.projection:
            # 1x1 Conv channel compression
            self.projection_layer = nn.Conv2d(refinenet_channels_in, 3, 1, 1)
            new_channels_in = 3
        else:
            new_channels_in = refinenet_channels_in

        if refinenet_conv_size == 5:
            self.conv = nn.Sequential(
                nn.Conv2d(new_channels_in, conv_channels, 5, 1, 2),
                nn.ELU(True),
                nn.Conv2d(conv_channels, conv_channels, 5, 1, 2),
                nn.ELU(True),
                nn.Conv2d(conv_channels, conv_channels, 5, 1, 2),
                nn.ELU(True),
                nn.AvgPool2d(4),
                Flatten(),
                nn.Linear((input_size[1]//4)*(input_size[1]//4)*conv_channels, lstm_dim),
                nn.ELU(True)
            )
        elif refinenet_conv_size == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(new_channels_in, conv_channels, 3, 1, 1),
                nn.ELU(True),
                nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
                nn.ELU(True),
                nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
                nn.ELU(True),
                nn.AvgPool2d(4),
                Flatten(),
                nn.Linear((input_size[1]//4)*(input_size[1]//4)*conv_channels, lstm_dim),
                nn.ELU(True)
            )


        self.input_proj = nn.Sequential(
                nn.Linear(lstm_dim + 4*self.z_size, lstm_dim),
                nn.ELU(True)
            )
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)
        self.loc = nn.Linear(lstm_dim, z_size)
        self.scale = nn.Linear(lstm_dim, z_size)


    def forward(self, img_inputs, vec_inputs, h, c):
        """
        img_inputs: [N * K, C, H, W]
        vec_inputs: [N * K, 4*z_size]
        """
        if self.projection:
            img_inputs = self.projection_layer(img_inputs)
        x = self.conv(img_inputs)
        # concat with \lambda and \nabla \lambda
        x = torch.cat([x, vec_inputs], 1)
        x = self.input_proj(x)
        x = x.unsqueeze(0) # seq dim
        # TODO: break deterministic path from input image to reconstruction
        # by stepping dynamics here
        self.lstm.flatten_parameters()
        out, (h,c) = self.lstm(x, (h,c))
        out = out.squeeze(0)
        loc = self.loc(out)
        scale = self.scale(out)
        lamda = torch.cat([loc, scale], 1)
        return lamda, (h,c)


class SpatialBroadcastDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets
    into RGB and mask
    """

    @net.capture
    def __init__(self, input_size, output_size, z_size, conv_channels):
        super(SpatialBroadcastDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        self.decode = nn.Sequential(
            nn.Conv2d(z_size+2, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, output_size, 1, 1)
        )
        self.output_image_channels = output_size-1

    @staticmethod
    def spatial_broadcast(z, h, w):
        """
        source: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py
        """

        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, z):
        z_sb = SpatialBroadcastDecoder.spatial_broadcast(z, self.h + 8, self.w + 8)
        out = self.decode(z_sb) # [batch_size * K, output_size, h, w]
        if self.output_image_channels == 3:
            return torch.sigmoid(out[:,:self.output_image_channels]), out[:,-1]
        else:
            return out[:,:self.output_image_channels], out[:,-1]


class ComponentEncoder(nn.Module):

    @net.capture
    def __init__(self, input_size, z_size):
        super(ComponentEncoder, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        M = 1
        self.encode = nn.Sequential(
            GatedConv2dBN(self.input_size[0], 32 * M, 1, 2),
            GatedConv2dBN(32*M, 32*M, 2, 2),
            GatedConv2dBN(32*M, 64*M, 1, 2),
            GatedConv2dBN(64*M, 64*M, 2, 2),
            GatedConv2dBN(64*M, 64*M, 1, 2),
            Flatten(),
            nn.Linear((self.input_size[1]//4)**2 * 64*M, 256),
            nn.ELU(True)
        )
        self.loc = nn.Linear(256, self.z_size)
        self.scale = nn.Linear(256, self.z_size)

    def forward(self, x):
        x = self.encode(x)
        loc = self.loc(x)
        scale = self.scale(x)
        return loc, scale


def refinenet_sequential_inputs(image, means, masks, mask_logits, log_p_k, normal_ll, lamda, loss, layer_norms, eval_mode):
    N, K, C, H, W = image.shape
    # 1. image [N, K, C, H, W]
    # 2. means [N, K, C, H, W]
    # 3. masks  [N, K, 1, H, W] (log probs)

    # 8. Coordinate channels
    x = torch.linspace(-1, 1, W).to(image.device)
    y = torch.linspace(-1, 1, H).to(image.device)
    y_b, x_b = torch.meshgrid(y, x)

    # Expand from (h, w) -> (n, k, 1, h, w)
    x_mesh = x_b.expand(N, K, 1, -1, -1)
    y_mesh = y_b.expand(N, K, 1, -1, -1)
    # 9. \partial L / \partial means
    # [N, K, C, H, W]
    # 10. \partial L/ \partial masks
    # [N, K, 1, H, W]
    # 11. \partial L/ \partial lamda
    # [N*K, 2 * self.z_size]
    d_means, d_masks, d_lamda = \
            torch.autograd.grad(loss, [means, masks, lamda], create_graph=not eval_mode,
                    retain_graph=not eval_mode, only_inputs=True)

    d_loc_z, d_var_z = d_lamda.chunk(2, dim=1)
    d_loc_z, d_var_z = d_loc_z.contiguous(), d_var_z.contiguous()

    # apply layernorms
    d_means = layer_norms[2](d_means).detach()
    d_masks = layer_norms[3](d_masks).detach()
    d_loc_z = layer_norms[4](d_loc_z).detach()
    d_var_z = layer_norms[5](d_var_z).detach()

    # concat image-size and vector inputs
    image_inputs = torch.cat([
        image, means, masks.exp(),
        d_means, d_masks, x_mesh, y_mesh], 2)
    
    vec_inputs = torch.cat([
        lamda, d_loc_z, d_var_z], 1)

    return image_inputs.view(N * K, -1, H, W), vec_inputs


class SDVAE(nn.Module):
    """
    iterative_inference_schedule: List of length T containing number of iterative inference steps to use per time step
    time_inference_schedule: List of length T specifying whether to use "L" (L.O.) or "M" (MoN) (e.g., ["L","M","L",...,"M"])

    """
    @net.capture
    def __init__(self, z_size, input_size, K, batch_size, log_scale, bg_log_scale,
            lstm_dim, iterative_inference_schedule, time_inference_schedule, kl_beta, stochastic_samples, 
            geco_warm_start, action_dim, action_conditional_dynamics, ssm, dynamics_uncertainty,
            context_len, iterative_VI_projection, separate_variances, action_shift, action_noise):
        super(SDVAE, self).__init__()
        self.context_len = context_len
        self.z_size = z_size
        self.input_size = input_size
        self.lstm_dim = lstm_dim
        self.K = K
        self.dynamics_uncertainty = dynamics_uncertainty
        self.iterative_inference_schedule = iterative_inference_schedule
        self.time_inference_schedule = time_inference_schedule
        self.separate_variances = separate_variances
        self.action_shift = action_shift
        self.action_noise = action_noise

        self.batch_size = batch_size
        self.stochastic_samples = stochastic_samples
        self.kl_beta = kl_beta
        self.gmm_log_scale = torch.cat([torch.FloatTensor([bg_log_scale]), log_scale * torch.ones(K-1)], 0)
        self.gmm_log_scale = self.gmm_log_scale.view(1, K, 1, 1, 1)

        self.geco_warm_start = geco_warm_start
        self.geco_C_ema = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.geco_beta = nn.Parameter(torch.tensor(0.55), requires_grad=False)
        
        self.refine_net = RefinementNetwork()
          
        if ssm == 'RSSM':
            decoder_input = 2 * z_size
            dynamics_recurrent_dim = z_size
        else:
            decoder_input = z_size
            dynamics_recurrent_dim = 2 * z_size

        self.image_decoder = SpatialBroadcastDecoder(z_size=decoder_input, output_size=input_size[0]+1)
        self.relational_dynamics = RelationalDynamics()

        init_weights(self.image_decoder, 'xavier')
        init_weights(self.refine_net, 'xavier')
        init_weights(self.relational_dynamics, 'xavier')
        
        self.foreground_K = self.K

        self.lamda_0 = nn.Parameter(torch.cat([\
                torch.zeros(1, self.z_size), torch.ones(1, self.z_size)],1))

        # layernorms for iterative inference input
        n = self.input_size[1]
        self.layer_norms = torch.nn.ModuleList([
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((3,n,n), elementwise_affine=False),
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((self.z_size,), elementwise_affine=False),
                nn.LayerNorm((self.z_size,), elementwise_affine=False)
            ])

        self.init_recurrent_states = {
            'n_step_dynamics': {
                'h': nn.Parameter(torch.zeros(1, self.K, dynamics_recurrent_dim)),
                },
            'inference_lambda': {}
        }
                    
    def get_action(self, seq_step, actions):
        if actions is not None:
            a_prev = actions[:,seq_step-self.action_shift]
            if self.action_noise != 0.:
                a_prev = a_prev + torch.zeros(a_prev.shape).normal_(0, self.action_noise).to(a_prev.device)
            return a_prev
        else:
            a_prev = None
        return a_prev

    def inference_step(self, x_t, geco, seq_step, global_step, posterior_zs, lambdas, current_recurrent_states, actions):
        total_loss = 0.
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]
        log_var = (2 * self.gmm_log_scale).to(x_t.device)
        dynamics_dist = []

        if len(self.iterative_inference_schedule) == 1:
            num_iters = self.iterative_inference_schedule[0]
        else:
            num_iters = self.iterative_inference_schedule[seq_step]
        

        if seq_step == 0:
            assert not torch.isnan(self.lamda_0).any(), 'lambda_0 has nan'
                # expand lambda_0
            lamda_0 = self.lamda_0.repeat(self.batch_size*self.foreground_K,1) # [N*K, 2*z_size]
            deterministic_state = current_recurrent_states['n_step_dynamics']['h']
            prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x_t.device)
        else:
            prior_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x_t.device)
            
            a_prev = self.get_action(seq_step, actions)
            
            if self.relational_dynamics.ssm == 'Ours':
                lamda_dynamics, q_h_dyn = self.relational_dynamics(torch.stack(posterior_zs), current_recurrent_states['n_step_dynamics']['h'], a_prev)
            elif self.relational_dynamics.ssm == 'SSM':
                lamda_dynamics, q_h_dyn = self.relational_dynamics(posterior_zs[-1].unsqueeze(0), current_recurrent_states['n_step_dynamics']['h'], a_prev)
            elif self.relational_dynamics.ssm == 'RSSM':
                z_prev = posterior_zs[-1]
                lamda_dynamics, q_h_dyn = self.relational_dynamics(z_prev, current_recurrent_states['n_step_dynamics']['h'], a_prev)
            # for next time step
            current_recurrent_states['n_step_dynamics']['h'] = q_h_dyn.unsqueeze(0)
            loc_z, var_z = lamda_dynamics.chunk(2, dim=1)
            loc_z, var_z = loc_z.contiguous(), var_z.contiguous()
            
            dynamics_prior_z = mvn(loc_z, var_z)
            dynamics_dist += [var_z.detach()]

            deterministic_state = current_recurrent_states['n_step_dynamics']['h']
            
            if self.separate_variances:
                lamda_0 = self.lamda_0.repeat(self.batch_size*self.K,1) # [N*K, 2*z_size]
                loc_z_, var_z_ = lamda_0.chunk(2, dim=1)
                loc_z_, var_z_ = loc_z_.contiguous(), var_z_.contiguous()
                
                # use the learned var shared across timesteps
                # and loc_z from dynamics
                dynamics_prior_z = mvn(loc_z, var_z_)
                # update lamda_dynamics
                lamda_0 = torch.cat([loc_z, var_z_],1)

            else:
                lamda_0 = lamda_dynamics

        h = current_recurrent_states['inference_lambda']['h']
        c = current_recurrent_states['inference_lambda']['c']
     
        for i in range(num_iters):
            loc_z, var_z = lamda_0.chunk(2, dim=1)
            loc_z, var_z = loc_z.contiguous(), var_z.contiguous()
            posterior_z = mvn(loc_z, var_z)
            detached_posterior_z = mvn(loc_z.detach(), var_z.detach())
            z = posterior_z.rsample()
            
            # Get means and masks based on SSM. RSSM adds the deterministic path from latent state to observation here.
            if self.relational_dynamics.ssm == 'RSSM':
                z_ = torch.cat([z, deterministic_state.view(-1, self.z_size)], 1)
                x_loc, mask_logits = self.image_decoder(z_)  #[N*K, C, H, W]
            elif self.relational_dynamics.ssm == 'Ours' or self.relational_dynamics.ssm == 'SSM':
                x_loc, mask_logits = self.image_decoder(z)  #[N*K, C, H, W]

            x_loc = x_loc.view(self.batch_size, self.K, C, H, W)

            # softmax across slots
            mask_logits = mask_logits.view(self.batch_size, self.K, 1, H, W)
            mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # NLL [batch_size, 1, H, W]
            nll, ll_outs = gmm_loglikelihood(x_t, x_loc, log_var, mask_logprobs)

            # KL div
            if seq_step == 0:
                kl_div = torch.distributions.kl.kl_divergence(posterior_z, prior_z)
                kl_div = kl_div.view(self.batch_size, self.K).sum(1)
                refine_foreground_only = False
            else:
                kl_div = torch.distributions.kl.kl_divergence(posterior_z, dynamics_prior_z)
                kl_div = kl_div.view(self.batch_size, self.K).sum(1)
                refine_foreground_only = False
            
            if self.geco_warm_start > global_step:
            #    # [batch_size]
                loss = torch.mean(nll + self.kl_beta * kl_div)
            else:
                loss = torch.mean(self.kl_beta * kl_div) - geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
            scaled_loss = ((i+1.) / num_iters) * loss
            total_loss += scaled_loss
            
            # Refinement
            if i == num_iters-1:
                # after T refinement steps, just output final loss
                #z_seq += [z]
                #break
                continue

            # compute refine inputs
            x_ = x_t.repeat(self.K, 1, 1, 1).view(self.batch_size, self.K, C, H, W)
            
            img_inps, vec_inps = refinenet_sequential_inputs(x_, x_loc, mask_logprobs,
                    mask_logits, ll_outs['log_p_k'], ll_outs['normal_ll'], lamda_0, loss, self.layer_norms,
                    not self.training)

            delta, (h,c) = self.refine_net(img_inps, vec_inps, h, c)
            lamda_0 = lamda_0 + delta
       
        posterior_zs += [z]
        lambdas += [lamda_0]

        return x_loc, mask_logprobs.exp(), nll, torch.mean(kl_div), posterior_zs, lambdas, total_loss, current_recurrent_states, i, dynamics_dist


    def rollout(self, latents, seq_step, current_recurrent_states, actions=None, x_t=None, compute_logprob=False):
        """
        z_history is List of length context_len of Tensors of shape [batch_size * K, z_size]
        
        rollout and decode for seq_len - context_len steps

        """
        a_prev = self.get_action(seq_step, actions)
        if a_prev is not None:
            a_prev = a_prev.repeat(self.stochastic_samples, 1)
        
        if self.relational_dynamics.ssm == 'Ours':
            lamda, h = self.relational_dynamics(torch.stack(latents), current_recurrent_states['n_step_dynamics']['h'], a_prev)
        elif self.relational_dynamics.ssm == 'SSM': 
            lamda, h = self.relational_dynamics(latents[-1].unsqueeze(0), current_recurrent_states['n_step_dynamics']['h'], a_prev)
        elif self.relational_dynamics.ssm == 'RSSM':
            latents_ = latents[-1]
            lamda, h = self.relational_dynamics(latents_, current_recurrent_states['n_step_dynamics']['h'], a_prev)
        h = h.unsqueeze(0)
        current_recurrent_states['n_step_dynamics']['h'] = h
       
        # TODO: this is actually _softplus_to_std(var_z) so change to softplus_std
        loc_z, var_z = lamda.chunk(2, dim=1)
        loc_z, var_z = loc_z.contiguous(), var_z.contiguous()
        p_z = mvn(loc_z, var_z)
        z_joint = p_z.rsample(torch.Size((1,)))
        z_joint = z_joint.view(self.stochastic_samples * self.batch_size * self.K, self.z_size)

        #if (decode_last and t == seq_len-1) or not decode_last:
        if self.relational_dynamics.ssm == 'RSSM':
            h_ = h.view(-1, self.z_size)
            z_joint_ = torch.cat([z_joint, h_],1)
            x_loc, mask_logits = self.image_decoder(z_joint_)
        elif self.relational_dynamics.ssm == 'Ours' or self.relational_dynamics.ssm == 'SSM':
            x_loc, mask_logits = self.image_decoder(z_joint)

        _, C, H, W = x_loc.shape
        x_loc = x_loc.view(self.stochastic_samples, self.batch_size, self.K, C, H, W)
        mask_logits = mask_logits.view(self.stochastic_samples * self.batch_size, self.K, 1, H, W)
        mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1).view(self.stochastic_samples, self.batch_size, self.K, 1, H, W)
        means = [x_loc.permute(1,0,2,3,4,5).contiguous()]
        masks = [torch.exp(mask_logprobs).permute(1,0,2,3,4,5).contiguous()]
        # add latents to sequence
        latents += [z_joint]

        if compute_logprob:
            log_var = (2 * self.gmm_log_scale).to(x_t.device)
            x_loc = x_loc.view(self.stochastic_samples * self.batch_size, self.K, C, H, W)
            mask_logprobs = mask_logprobs.view(self.stochastic_samples * self.batch_size, self.K, 1, H, W)
            # NLL [samples * batch_size]
            nll, _ = gmm_loglikelihood(x_t, x_loc, log_var, mask_logprobs)
            nll = nll.view(self.stochastic_samples, self.batch_size)
            nll_discounted = nll * (1 / ((seq_step - self.context_len)+1.))
            return means, masks, latents, current_recurrent_states, nll, nll_discounted, [var_z]

        return means, masks, latents, current_recurrent_states, [var_z]


    def dynamics_step(self, x_t, geco, seq_step, global_step, z_seq, current_recurrent_state, actions):
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]
        log_var = (2 * self.gmm_log_scale).to(x_t.device)
        
        a_prev = self.get_action(seq_step, actions)

        if self.relational_dynamics.ssm == 'Ours':
            lamda, h = self.relational_dynamics(torch.stack(z_seq), current_recurrent_state['n_step_dynamics']['h'], a_prev)
        elif self.relational_dynamics.ssm == 'SSM':
            lamda, h = self.relational_dynamics(z_seq[-1].unsqueeze(0), current_recurrent_state['n_step_dynamics']['h'], a_prev)
        elif self.relational_dynamics.ssm == 'RSSM':
            z_prev = z_seq[-1]  # [N*K, z_size]
            # [N*K, 2*z_size]
            lamda, h = self.relational_dynamics(z_prev, current_recurrent_state['n_step_dynamics']['h'], a_prev)
        current_recurrent_state['n_step_dynamics']['h'] = h.unsqueeze(0)

        loc_z, var_z = lamda.chunk(2, dim=1)
        loc_z, var_z = loc_z.contiguous(), var_z.contiguous()

        p_z = mvn(loc_z, var_z)

        z = p_z.rsample(torch.Size((1,)))
        z = z.view(self.batch_size * self.K, self.z_size)
        if self.relational_dynamics.ssm == 'RSSM':
            z_ = torch.cat([z, current_recurrent_state['n_step_dynamics']['h'].view(-1, self.z_size)], 1)
        elif self.relational_dynamics.ssm == 'Ours' or self.relational_dynamics.ssm == 'SSM':
            z_ = z

        x_loc, mask_logits = self.image_decoder(z_)  #[N*K, C, H, W]
        x_loc = x_loc.view(self.batch_size, self.K, C, H, W)
        mask_logits = mask_logits.view(self.batch_size, self.K, 1, H, W)
        mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1).view(self.batch_size, self.K, 1, H, W)
            
        # NLL [batch_size, 1, H, W]
        nll, _ = gmm_loglikelihood(x_t, x_loc, log_var, mask_logprobs)
        if self.geco_warm_start > global_step:
        #    # [batch_size]
            loss = torch.mean(nll)
        else:
            loss = -geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))

        z_seq += [z]

        return x_loc, mask_logprobs.exp(), nll, z_seq, current_recurrent_state, loss, [var_z]
    
    def dynamics_bms_step(self, x_t, geco, seq_step, global_step, z_seq, current_recurrent_state, actions):
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]
        log_var = (2 * self.gmm_log_scale).to(x_t.device)
        
        a_prev = self.get_action(seq_step, actions)

        if self.relational_dynamics.ssm == 'Ours':
            lamda, h = self.relational_dynamics(torch.stack(z_seq), current_recurrent_state['n_step_dynamics']['h'], a_prev)
        elif self.relational_dynamics.ssm == 'SSM':
            lamda, h = self.relational_dynamics(z_seq[-1].unsqueeze(0), current_recurrent_state['n_step_dynamics']['h'], a_prev)
        elif self.relational_dynamics.ssm == 'RSSM':
            z_prev = z_seq[-1]  # [N*K, z_size]
            # [N*K, 2*z_size]
            lamda, h = self.relational_dynamics(z_prev, current_recurrent_state['n_step_dynamics']['h'], a_prev)
        current_recurrent_state['n_step_dynamics']['h'] = h.unsqueeze(0)

        loc_z, var_z = lamda.chunk(2, dim=1)
        loc_z, var_z = loc_z.contiguous(), var_z.contiguous()

        p_z = mvn(loc_z, var_z)

        z = p_z.rsample(torch.Size((self.stochastic_samples,)))
        z = z.view(self.stochastic_samples * self.batch_size * self.K, self.z_size)
        
        x_t = x_t.repeat(self.stochastic_samples, 1, 1, 1)
        x_loc, mask_logits = self.image_decoder(z)  #[N*K, C, H, W]
        x_loc = x_loc.view(self.stochastic_samples * self.batch_size, self.K, C, H, W)
        mask_logits = mask_logits.view(self.stochastic_samples * self.batch_size, self.K, 1, H, W)
        mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1).view(self.stochastic_samples * self.batch_size, self.K, 1, H, W)
            
        # NLL [stochastic_samples * batch_size, 1, H, W]
        nll, _ = gmm_loglikelihood(x_t, x_loc, log_var, mask_logprobs)
        nll = nll.view(self.stochastic_samples, self.batch_size)
        best = torch.argmax(-nll, 0)
        sample_idxs = torch.arange(self.batch_size).to(x_t.device) * self.stochastic_samples + best
    
        nll = nll.permute(1,0).contiguous().view(-1)
        nll = nll[sample_idxs]

        z = z.view(-1, self.K, self.z_size)
        z = z[sample_idxs]
        z = z.view(self.batch_size * self.K, self.z_size)

        x_loc = x_loc.view(self.stochastic_samples, self.batch_size, self.K, C, H, W).permute(1,0,2,3,4,5).contiguous()
        x_loc = x_loc.view(-1, self.K, C, H, W)
        x_loc = x_loc[sample_idxs]
        
        mask_logprobs = mask_logprobs.view(self.stochastic_samples, self.batch_size, self.K, 1, H, W).permute(1,0,2,3,4,5).contiguous()
        mask_logprobs = mask_logprobs.view(-1, self.K, 1, H, W)
        mask_logprobs = mask_logprobs[sample_idxs]

        if self.geco_warm_start > global_step:
        #    # [batch_size]
            loss = torch.mean(nll)
        else:
            loss = -geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
        z_seq += [z]
        
        
        return x_loc, mask_logprobs.exp(), nll, z_seq, current_recurrent_state, loss, [var_z]
    
    
    def uncertainty_loss(self, nll_t, geco, global_step):
        """
        List of length sequence length of nlls of shape [samples, batch_size]
        """
        logprobs = -torch.stack(nll_t)  # [T, S, N]
        T = logprobs.shape[0]
        logprobs = torch.sum(logprobs, 0)  # [S, N]
        max_logprobs, max_indices = torch.max(logprobs, dim=0)  # [N]
        nll = -max_logprobs
        if self.geco_warm_start <= global_step:
            loss = -geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll) / T)
        else:
            loss = nll
        return loss, max_indices, nll

    
    def forward(self, x, actions, geco, global_step):
        """
        x: [batch_size, T, C, H, W]
        """
        x_means, masks, nll_t, kl_t, prior_zs, posterior_zs, lambdas, inference_steps = [], [], [], [], [], [], [], []
        rollout_nll = []
        undiscounted_nll = []
        dynamics_dist = []

        total_losses = 0.

        current_recurrent_states = {
            'n_step_dynamics': {
                'h': self.init_recurrent_states['n_step_dynamics']['h'].to(x.device).repeat(1, self.batch_size, 1)
                }
        }

        for step_index, inference_step in enumerate(self.time_inference_schedule):
            
            current_recurrent_states['inference_lambda'] = {
                'h': torch.zeros(1, self.batch_size * self.K, self.lstm_dim).to(x.device),
                'c': torch.zeros(1, self.batch_size * self.K, self.lstm_dim).to(x.device)}
            

            if inference_step == "I":
                x_loc, mask, nll, kl, posterior_zs, lambdas, total_loss, current_recurrent_states, i, dyn_dist = self.inference_step(
                        x[:,step_index], geco, step_index, global_step, posterior_zs, lambdas, current_recurrent_states, actions)
                x_means += [x_loc]
                masks += [mask]
                nll_t += [nll]
                kl_t += [kl]
                inference_steps += [i]
                dynamics_dist += dyn_dist
                total_losses += total_loss

            # Dynamics
            elif inference_step == "D":
                x_t = x[:, step_index]

                x_loc, mask, nll, posterior_zs, current_recurrent_states, loss, dyn_dist = \
                        self.dynamics_step(x_t, geco, step_index, global_step, posterior_zs, current_recurrent_states, actions)
                x_means += [x_loc]
                masks += [mask]
                nll_t += [nll]
                total_losses += loss
                dynamics_dist += dyn_dist
            
            # Dynamics BMS
            elif inference_step == "BMS":
                x_t = x[:, step_index]

                x_loc, mask, nll, posterior_zs, current_recurrent_states, loss, dyn_dist = \
                        self.dynamics_bms_step(x_t, geco, step_index, global_step, posterior_zs, current_recurrent_states, actions)
                x_means += [x_loc]
                masks += [mask]
                nll_t += [nll]
                total_losses += loss
                dynamics_dist += dyn_dist

            # Random rollout
            elif inference_step == "R":
                with torch.no_grad():
                    if step_index == self.context_len:
                        z_prev = torch.stack(posterior_zs)
                        # copy z_prev to [context_len, n_samples*batch_size*K, z_size]
                        z_prev = z_prev.repeat(1, self.stochastic_samples, 1)
                        z_prev = list(torch.chunk(z_prev, self.context_len, dim=0))  # list of [n_samples*N*K, z_size]
                        posterior_zs = [_.squeeze(0) for _ in z_prev]
                        current_recurrent_states['n_step_dynamics']['h'] = current_recurrent_states['n_step_dynamics']['h'].to(x.device).repeat(1, self.stochastic_samples, 1)
                    x_loc, mask, posterior_zs, current_recurrent_states, dyn_dist = self.rollout(posterior_zs, step_index, current_recurrent_states, actions)
                    if x_means[-1].shape[1] != self.stochastic_samples:
                        x_means = [_.unsqueeze(1).repeat(1, self.stochastic_samples, 1, 1, 1, 1) for _ in x_means]
                        masks = [m.unsqueeze(1).repeat(1, self.stochastic_samples, 1, 1, 1, 1) for m in masks]
                    x_means = x_means + x_loc 
                    masks = masks + mask
                    dynamics_dist += dyn_dist
            elif inference_step == "U":  # O(TSNK) complexity
                if step_index == self.context_len:
                    z_prev = torch.stack(posterior_zs)
                    # copy z_prev to [context_len, n_samples*batch_size*K, z_size]
                    z_prev = z_prev.repeat(1, self.stochastic_samples, 1)
                    z_prev = list(torch.chunk(z_prev, self.context_len, dim=0))  # list of [n_samples*N*K, z_size]
                    posterior_zs = [_.squeeze(0) for _ in z_prev]
                    current_recurrent_states['n_step_dynamics']['h'] = current_recurrent_states['n_step_dynamics']['h'].to(x.device).repeat(1, self.stochastic_samples, 1)
                    x_means = [_.unsqueeze(0).repeat(self.stochastic_samples, 1, 1, 1, 1, 1).permute(1,0,2,3,4,5).contiguous() for _ in x_means]
                    masks = [_.unsqueeze(0).repeat(self.stochastic_samples, 1, 1, 1, 1, 1).permute(1,0,2,3,4,5).contiguous() for _ in masks]
                
                x_t = x[:, step_index].repeat(self.stochastic_samples, 1, 1, 1)
                x_loc, mask, posterior_zs, current_recurrent_states, nll, nll_disc, _ = self.rollout(posterior_zs, step_index, current_recurrent_states, actions, x_t, True)
                x_means = x_means + x_loc 
                masks = masks + mask
                
                rollout_nll += [nll]
                #undiscounted_nll += [nll]
                
        if self.dynamics_uncertainty:
            _, _, C, H, W = x.shape
            loss, best_indices, best_nll = self.uncertainty_loss(rollout_nll, geco, global_step)
            total_losses += torch.mean(loss)  # average over batch size (summed over rollout steps)
            best_x_means, best_masks = [], []
            batch_ids = torch.arange(self.batch_size).to(x.device)
            best_batch_indices = (self.stochastic_samples * batch_ids) + best_indices

            for i in range(len(x_means)):
                x_m = x_means[i].view(self.stochastic_samples, self.batch_size, self.K, C, H, W)
                x_m = x_m.view(-1, self.K, C, H, W)
                
                masks_m = masks[i].view(self.stochastic_samples, self.batch_size, self.K, 1, H, W)
                masks_m = masks_m.view(-1, self.K, 1, H, W)

                best_x_means += [x_m[best_batch_indices]]
                best_masks += [masks_m[best_batch_indices]]

            x_means = best_x_means
            masks = best_masks
            nll_to_return = torch.sum(torch.mean(torch.stack(nll_t), dim=1)) + torch.mean(best_nll)
        else:
            nll_to_return = torch.sum(torch.mean(torch.stack(nll_t), dim=1))

        outs = {
            'total_loss': total_losses,
            'nll': nll_to_return,
            'kl': torch.sum(torch.stack(kl_t)),
            'x_means': x_means,
            'masks': masks,
            'posterior_zs': posterior_zs,
            'inference_steps': torch.mean(torch.Tensor(inference_steps).to(x.device)),
            'dynamics': dynamics_dist,
            'lambdas': lambdas
        }
        return outs



class VRNN(nn.Module):
    
    @net.capture
    def __init__(self, z_size, batch_size, context_len, geco_warm_start, input_size,
        log_scale, stochastic_samples, kl_beta, action_dim, action_shift, action_noise):
        super(VRNN, self).__init__()
        self.hidden_dim=256
        self.z_size = z_size
        self.batch_size = batch_size
        self.context_len = context_len
        self.input_size = input_size
        self.log_scale = torch.FloatTensor([log_scale])
        self.stochastic_samples = stochastic_samples
        self.kl_beta = kl_beta
        self.action_dim = action_dim
        self.action_shift = action_shift
        self.action_noise = action_noise
        self.time_inference_schedule = None
        self.dynamics_uncertainty=False

        self.geco_warm_start = geco_warm_start
        self.geco_C_ema = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.geco_beta = nn.Parameter(torch.tensor(0.55), requires_grad=False)
        
        self.phi_x = nn.Sequential(
            GatedConv2dBN(self.input_size[0], 32, 1, 2),
            GatedConv2dBN(32, 32, 2, 2),
            GatedConv2dBN(32, 64, 1, 2),
            GatedConv2dBN(64, 64, 2, 2),
            GatedConv2dBN(64, 64, 1, 2),
            Flatten(),
            nn.Linear((self.input_size[1]//4)**2 * 64, self.hidden_dim),
            nn.ELU(True),
        )
        init_weights(self.phi_x, 'xavier')
        
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_size, self.hidden_dim),
            nn.ELU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(True)
        )
        init_weights(self.phi_z, 'xavier')

        self.encoder = nn.Sequential(
            nn.Linear(self.action_dim + self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.ELU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(True)
        )
        init_weights(self.encoder, 'xavier')

        self.q_loc = nn.Linear(self.hidden_dim, self.z_size)
        self.q_var= nn.Linear(self.hidden_dim, self.z_size)
        init_weights(self.q_loc, 'xavier')
        init_weights(self.q_var, 'xavier')

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.action_dim + self.hidden_dim, self.hidden_dim),
            nn.ELU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(True)
        )
        init_weights(self.prior, 'xavier')

        self.p_loc = nn.Linear(self.hidden_dim, self.z_size)
        self.p_var = nn.Linear(self.hidden_dim, self.z_size)
        init_weights(self.p_loc, 'xavier')
        init_weights(self.p_var, 'xavier')

        # decoder
        self.dec = SpatialBroadcastDecoder(z_size=(self.hidden_dim * 2), output_size=4)
        init_weights(self.dec, 'xavier')

        # recurrence
        self.lstm = nn.LSTM(self.hidden_dim * 2, self.hidden_dim)

        self.h_0, self.c_0 = (torch.zeros(1, 1, self.hidden_dim),
                    torch.zeros(1, 1, self.hidden_dim))


    def freeze_inference(self):
        pass


    def get_action(self, seq_step, actions):
        if actions is not None:
            a_prev = actions[:,seq_step-self.action_shift]
            if self.action_noise != 0.:
                #if a_prev.shape[-1] == 6: # 3D pick and place
                #    a_old = a_prev
                #    a_prev = a_prev + torch.zeros(a_prev.shape).normal_(0, self.action_noise).to(a_prev.device)
                #    a_prev[:,3:] = a_old[:,3:] # old place
                #else:    
                a_prev = a_prev + torch.zeros(a_prev.shape).normal_(0, self.action_noise).to(a_prev.device)
            return a_prev
        else:
            a_prev = None
        return a_prev
    
    def rollout(self, h, c, seq_len, actions=None):
        """
        z_history is List of length context_len of Tensors of shape [batch_size * K, z_size]
        
        rollout and decode for seq_len - context_len steps
        actions is [seq_len, action_dim]

        """
        means = []
        masks = []
        for t in range(self.context_len, seq_len):
            if t == self.context_len:
                #h = h.repeat(1, self.stochastic_samples, 1)  # [1, batch_size * stochastic_samples, hidden_dim]
                #c = c.repeat(1, self.stochastic_samples, 1)  # [1, batch_size * stochastic_samples, hidden_dim]
                h = h.unsqueeze(2).repeat(1,1,self.stochastic_samples,1)  # [1, batch_size, self.stochastic_samples, hidden_dim]
                h = h.view(1, self.batch_size * self.stochastic_samples, -1)
                c = c.unsqueeze(2).repeat(1,1,self.stochastic_samples,1)  # [1, batch_size, self.stochastic_samples, hidden_dim]
                c = c.view(1, self.batch_size * self.stochastic_samples, -1)
                if actions is not None:
                    actions = actions.unsqueeze(1).repeat(1, self.stochastic_samples, 1, 1)
                    actions = actions.view(self.batch_size * self.stochastic_samples, -1, self.action_dim)  # [batch * stochastic_samples, seq_len, 4]

            a_t = self.get_action(t, actions) 
            if a_t is not None:
                # prior
                p_t = self.prior(torch.cat([a_t, h.squeeze(0)], 1))
            else:
                p_t = self.prior(h.squeeze(0))
                
            p_t_loc = self.p_loc(p_t)
            p_t_var= self.p_var(p_t)
            p_t = mvn(p_t_loc, p_t_var)
            
            # sample
            z_t = p_t.sample()
            phi_z_t = self.phi_z(z_t)

             # decode 
            x_means, _ = self.dec(torch.cat([phi_z_t, h.squeeze(0)], 1))
            _, C, H, W = x_means.shape
            # autoregressively encode own output
            phi_x_t = self.phi_x(x_means)

            # recurrence
            self.lstm.flatten_parameters()
            out, (h,c) = self.lstm(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h,c))
            #means += [x_means.view(self.stochastic_samples, self.batch_size, C, H, W).permute(1,0,2,3,4).contiguous()]
            means += [x_means.view(self.batch_size, self.stochastic_samples, C, H, W)]
            masks += [x_means[:,0]]

        return means, masks  # List of length seq_len - context_len of images and masks


    def forward(self, x, actions, geco, step):
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]
        T = x.shape[1]
        total_loss = 0.

        x_means_t = []
        masks_t = []  # empty
        nll_t = []
        kl_t = []

        h, c = self.h_0, self.c_0
        h = h.to(x.device).repeat(1, self.batch_size, 1)
        c = c.to(x.device).repeat(1, self.batch_size, 1)
        log_var = (2 * self.log_scale).to(x.device)
        
        if self.training:
            context_len = T
        else:
            context_len = self.context_len

        for t in range(context_len):
            x_t = x[:,t]
            a_t = self.get_action(t, actions)

            phi_x_t = self.phi_x(x_t)
            # encoder
            if a_t is not None:
                enc_t = self.encoder(torch.cat([a_t, phi_x_t, h.squeeze(0)], 1))
            else:
                enc_t = self.encoder(torch.cat([phi_x_t, h.squeeze(0)], 1))

            q_t_loc = self.q_loc(enc_t)
            q_t_var = self.q_var(enc_t)
            q_t = mvn(q_t_loc, q_t_var)

            # prior
            # N.b. SVG-LP provides phi_x_t-1 to the prior from ground truth x_t-1
            # VRNN passes phi_x_t-1 into the LSTM which gets processed by prior at time t "h", 
            # which gives p_t. Same!
            if a_t is not None:
                # prior
                p_t = self.prior(torch.cat([a_t, h.squeeze(0)], 1))
            else:
                p_t = self.prior(h.squeeze(0))
            p_t_loc = self.p_loc(p_t)
            p_t_var= self.p_var(p_t)
            p_t = mvn(p_t_loc, p_t_var)

            # sample
            z_t = q_t.rsample()
            phi_z_t = self.phi_z(z_t)

            # decode
            x_means, _ = self.dec(torch.cat([phi_z_t, h.squeeze(0)], 1))

            # recurrence
            self.lstm.flatten_parameters()
            out, (h,c) = self.lstm(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h,c))

            # Loss
            # KL
            kl_div = torch.distributions.kl.kl_divergence(q_t, p_t)  #[batch_size]
            # NLL
            nll = gaussian_loglikelihood(x_t, x_means, log_var)

            if self.geco_warm_start > step:
                loss = torch.mean(nll + self.kl_beta * kl_div)
            else:
                loss = torch.mean(self.kl_beta * kl_div) - \
                    geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
            total_loss += loss

            x_means_t += [x_means]
            masks_t += [x_means[:,0]]
            nll_t += [torch.mean(nll)]
            kl_t += [torch.mean(kl_div)]
        
        if not self.training:
            with torch.no_grad():
                pred_x_means, pred_x_masks = self.rollout(h, c, T, actions)
                x_means_t = [_.unsqueeze(1).repeat(1, self.stochastic_samples, 1, 1, 1) for _ in x_means_t]
                masks_t = [_.unsqueeze(1).repeat(1, self.stochastic_samples, 1, 1, 1) for _ in masks_t]
                x_means_t = x_means_t + pred_x_means
                masks_t = masks_t + pred_x_masks

                return {
                    'x_means': x_means_t,
                    'masks': masks_t
                }

        for t in range(context_len, T):
            x_t = x[:,t]
            a_t = self.get_action(t, actions)
            # prior
            if a_t is not None:
                p_t = self.prior(torch.cat([a_t,h.squeeze(0)],1))
            else:
                p_t = self.prior(torch.cat([h.squeeze(0)],1))

            p_t_loc = self.p_loc(p_t)
            p_t_var= self.p_var(p_t)
            p_t = mvn(p_t_loc, p_t_var)

            # sample
            z_t = p_t.rsample()
            phi_z_t = self.phi_z(z_t)

             # decode
            x_means, _ = self.dec(torch.cat([phi_z_t, h.squeeze(0)], 1))
            # autoregressively encode own output
            phi_x_t = self.phi_x(x_means)

            # recurrence
            self.lstm.flatten_parameters()
            out, (h,c) = self.lstm(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h,c))

            # Loss
            # NLL
            nll = gaussian_loglikelihood(x_t, x_means, log_var)

            if self.geco_warm_start > step:
                loss = torch.mean(nll)
            else:
                loss = -geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
            total_loss += loss

            x_means_t += [x_means]
            masks_t += [x_means[:,0]]
            nll_t += [torch.mean(nll)]

        return {
            'total_loss': total_loss,
            'nll': torch.sum(torch.stack(nll_t)),
            'kl': torch.sum(torch.stack(kl_t)),
            'x_means': x_means_t,
            'masks': masks_t,
            'inference_steps': torch.zeros(1).to(x.device)
        }
