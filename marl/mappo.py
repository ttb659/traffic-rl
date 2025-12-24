import torch
import torch.nn.functional as F


class MAPPO:
    def __init__(self, actor, critic, gnn, lr=3e-4, gamma=0.99, clip=0.2):
        self.actor = actor
        self.critic = critic
        self.gnn = gnn

        self.gamma = gamma
        self.clip = clip

        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) +
            list(critic.parameters()) +
            list(gnn.parameters()),
            lr=lr
        )

    """def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)"""

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0
        values = values + [0.0]  # bootstrap final V(T+1) = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)


    def update(self, batch):
        """
        batch contient :
        - states: (T, N, obs_dim)
        - actions: (T, N)
        - log_probs: (T, N)
        - rewards: (T,)
        """

        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        rewards = batch["rewards"]

        T, N, _ = states.shape

        #### returns = self.compute_returns(rewards)

        # GNN encoding par pas de temps
        embeddings = []
        for t in range(T):
            h_t = self.gnn(states[t], batch["adj"]) #(N, emb_dim)
            embeddings.append(h_t)
        
        embeddings = torch.stack(embeddings) # (T, N, emb_dim)

        # Critic
        """values = torch.stack([
            self.critic(embeddings[t]) for t in range(T)
        ]).squeeze()

        advantages = returns - values.detach()"""

        # =====================
        # CRITIC VALUES
        # =====================
        values = torch.stack([
            self.critic(embeddings[t]) for t in range(T)
        ]).squeeze()

        # =====================
        # GAE
        # =====================
        advantages = self.compute_gae(
            rewards,
            values.detach().tolist(),
            gamma=self.gamma,
            lam=0.95
        )

        returns = advantages + values.detach()

        # Normalisation des advantages (très important)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)



        # Actor loss
        dist = self.actor(embeddings.view(-1, embeddings.shape[-1]))
        new_log_probs = dist.log_prob(actions.view(-1)).view(T, N)

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages.unsqueeze(1)

        for _ in range(4):
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().mean()
            actor_loss = actor_loss - 0.01 * entropy


        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
