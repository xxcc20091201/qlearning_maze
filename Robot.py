import random
import numpy as np
class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0
        
        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        #print('Robot reset: ')
        self.state = self.maze.sense_robot()
        self.create_Qtable_line(self.state)
        self.t = self.t + 1

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 0
            pass
        else:
            # TODO 2. Update parameters when learning
            self.epsilon = self.epsilon0/self.t
            pass

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        state = self.state
        #state = self.maze.sense_robot()
        #print('Robot sense state: ', state)
        return state

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        #print('create_Qtable_line, Qtable:',self.Qtable )
        
        if state not in self.Qtable:
            #print(state, 'is not in Qtable')
            self.Qtable[state] = {'u': 0, 'd': 0,'l': 0,'r': 0}
        #print('After create_Qtable_line, new Qtable:',self.Qtable ,'state:', self.state)
        
        pass

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            tmp = random.random()
            #print(tmp,'  ',self.epsilon)
            if ( self.epsilon > tmp or abs(self.epsilon - tmp) < 0.4) :
                return True
            else:
                return False
            pass
        
        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                #print('exploration')
                action = self.valid_actions[random.randint(0,3)]
                #return None
            else:
                # TODO 7. Return action with highest q value
                #print('qline',self.Qtable,'self.state',self.state, 'self.sense_robot', self.maze.sense_robot())
                qline = self.Qtable[self.state]
                action = max(qline, key = qline.get)
                #return None
        elif self.testing:    
            # TODO 7. choose action with highest q value
            qline = self.Qtable[self.state]
            action = max(qline, key = qline.get)
        else:
            # TODO 6. Return random choose aciton
            action = self.valid_actions[random.randint(0,3)]
        #print('choose action:', action)
        self.t = self.t + 1
        return action
            
    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            
            # TODO 8. When learning, update the q table according
            # to the given rules
            #print('Robot next_state', next_state)
            #print('Robot next_state --> Qtable[next_state]:', self.Qtable[next_state])
            #print('update_Qtable, next_state:', next_state, 'self.state:', self.state)
            max_key = max(self.Qtable[next_state], key = self.Qtable[next_state].get)
            #print('max_key: ', max_key)
            #print('max_key --> self.Qtable[next_state][max_key]:', self.Qtable[next_state][max_key])
            #print('alpha:', self.alpha, 'reword:', r)
            #print('self.Qtable[self.state][action]:',self.Qtable[self.state][action], 'action:',action)
            self.Qtable[self.state][action] = self.Qtable[self.state][action] + self.alpha * (r + self.gamma * self.Qtable[next_state][max_key] - self.Qtable[self.state][action])
            
            #print('updated Qtable(update_Qtable):', self.Qtable)
            pass
        
    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        #print('update begin: self.state', self.state)
        self.state = self.maze.sense_robot() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line
        #print('update choose_action')
        
        #print('Before move robot, state:', self.state, 'maze.sense_robot:', self.maze.sense_robot())
        action = self.choose_action() # choose action for this state
        #self.state = self.maze.sense_robot()
        reward = self.maze.move_robot(action) # move robot for given action
        #print('After move robot, state:', self.state, 'maze.sense_robot:', self.maze.sense_robot())
        
        #print('update  next_state:')
        
        next_state = self.maze.sense_robot() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
