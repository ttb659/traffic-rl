import torch
import torch.nn.functional as F
import os

class MAPPO:
    def __init__(
        self,
        actor,
        critic,
        gnn,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        gae_lambda=0.95,
        ppo_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5
    ):
        self.actor = actor
        self.critic = critic
        self.gnn = gnn

        self.gamma = gamma
        self.clip = clip
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) +
            list(critic.parameters()) +
            list(gnn.parameters()),
            lr=lr
        )

    # ======================================================
    # GAE
    # ======================================================
    def compute_gae(self, rewards, values, dones):
        """
        rewards : (T,)
        values  : (T+1,)
        dones   : (T,)
        """
        advantages = torch.zeros(len(rewards))
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    # ======================================================
    # UPDATE MAPPO 
    # ======================================================
    def update(self, batch):
        states = batch["states"]       # (T, N, obs_dim)
        actions = batch["actions"]     # (T, N)
        old_log_probs = batch["log_probs"]  # (T, N)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32)  # (T,)
        dones = torch.zeros(len(rewards))  # épisodes tronqués (OK)

        T, N, _ = states.shape

        # ==================================================
        # GNN ENCODING (T, N, emb_dim)
        # ==================================================
        embeddings = []
        for t in range(T):
            h_t = self.gnn(states[t], batch["adj"])
            embeddings.append(h_t)

        embeddings = torch.stack(embeddings).detach()

        # ==================================================
        # CRITIC VALUES (centralized)
        # ==================================================
        values = torch.stack([
            self.critic(embeddings[t]) for t in range(T)
        ]).squeeze()

        # Bootstrap V(T+1) = 0 (fin épisode)
        values = torch.cat([values, torch.zeros(1)])

        # ==================================================
        # GAE
        # ==================================================
        advantages, returns = self.compute_gae(
            rewards,
            values.detach(),
            dones
        )

        # Normalisation des advantages (TRÈS IMPORTANT)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==================================================
        # PPO UPDATE (MULTI-EPOCH RÉEL)
        # ==================================================
        for _ in range(self.ppo_epochs):
            # Actor
            dist = self.actor(embeddings.view(-1, embeddings.shape[-1]))
            new_log_probs = dist.log_prob(actions.view(-1)).view(T, N)

            ratio = torch.exp(new_log_probs - old_log_probs)

            adv = advantages.unsqueeze(1)  # broadcast (T, N)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            # Critic
            value_preds = torch.stack([
                self.critic(embeddings[t]) for t in range(T)
            ]).squeeze()

            critic_loss = F.mse_loss(value_preds, returns)

            loss = (
                actor_loss
                + self.value_coef * critic_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # --- LOG VALUES ---
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_entropy = entropy.item()
    

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "gnn": self.gnn.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.gnn.load_state_dict(checkpoint["gnn"])

