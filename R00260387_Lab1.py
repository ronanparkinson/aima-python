# Theoretical exercise answers

# 1. It's rational as it's expected performance would be as good as any other agents performence as it will be able to check that the location that it knows and understands for dirt.
# It will be able to percive if the location needs to be cleaned or if the location it is in is clean and can move to the other known location in the world it is perciving. 

import agents
from agents import compare_agents

def TrivialVacuumAgent():

    def execute_action(self, agent, action):

        if action == 'Right':
            agent.location = agents.loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = agents.loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    return agents.Agent(execute_action)



# 2. For a rational agent to perform as expected in this senario, the agent will need to decide what is the best action to take in it's current state ( location and is the loction clean or 
# dirty). Assuming that agent knows the environment, it will know if the two location as are dirty or clean after visting each loction. A rational movement would be to stop moving until it 
# becomes aware that one of the locations becomes dirty. 
# It could have a timer to check each location for dirty or stop the agent from moving relentlessly when all areas are clean.
# 
# 3. It could have a timer to check each location for dirty or stop the agent from moving relentlessly when all areas are clean. 
# An agent would need to be allowed to expore before becoming familaiar with the enviroment. It should learn how many locations are in the enviroment and what locations are 
# dirty. If it is not left explore, it will not monitor all of the loctions in the environment. We are trying to get the agent to be rational in the enviroment, try to have an 
# omniscience agent as this is not realistic is real world scenarios. 





