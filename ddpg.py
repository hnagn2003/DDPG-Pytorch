from models.backbone import Actor, Critic
from torch.optim import Adam
from utils import OURandomProcess
class DDPG():
    def __init__(self, state_size, action_size, args) -> None:
        if args.seed > 0:
            self.seed(args.seed)
        self.batch_size = args.batch_size
        self.state_size = state_size
        self.action_size= action_size
        self.tau = args.tau # soft update factor
        self.discount = args.discount
        self.reciprocal = 1.0 / args.epsilon
        self.epsilon = 1.0
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.state_size, self.action_size, **net_cfg)
        self.actor_target = Actor(self.state_size, self.action_size, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.state_size, self.action_size, **net_cfg)
        self.critic_target = Critic(self.state_size, self.action_size, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        
        self.actor_target = self.actor
        self.critic_target = self.critic
        
        # TODO experience buffer
        # TODO eval
        # TODO (optionals) cuda, load save weights,...
        
def update(self):
    # sample
    # next q 
    # update critic
    # update actor
    # soft update target