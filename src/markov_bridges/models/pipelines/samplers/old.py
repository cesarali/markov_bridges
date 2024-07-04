"""
def S4_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='None', corrector='None', 
                        snr=0.1, scale_eps=1.0, n_steps=1, 
                        probability_flow=False, continuous=False,
                        denoise=True, eps=1e-3, device='cuda'):

  def s4_solver(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      dt = -1. / diff_steps

      # -------- Rverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t
        vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2) 

        # -------- Score computation --------
        score_x = score_fn_x(x, adj, flags, vec_t)
        score_adj = score_fn_adj(x, adj, flags, vec_t)

        Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
        Sdrift_adj  = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

        # -------- Correction step --------
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)
      
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * score_x
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * score_adj
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        # -------- Prediction step --------
        x_mean = x
        adj_mean = adj
        mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
        
        x = x + Sdrift_x * dt
        adj = adj + Sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt) 
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

        x_mean = mu_x
        adj_mean = mu_adj
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), 0
  return s4_solver
"""