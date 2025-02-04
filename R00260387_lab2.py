#A. 2)
#I would use a table stae diagram as it will be easy to see the record kept of all of the different actions taken in the scenario and will eventually come out with the correct solution listed in the table. 

# 3 + 4)
# Starting off in location A - picks up the chicken and move the chicken to location B - Moves back to location A -
# Picks up fox - moves fox to location B - Picks up chicken from location B - Moves chicken to loction A - 
# Picks up feed at loction A - moves feed to location B - Goes back to location A - picks up chicken and moves the chicken to loction B

# 5)
import agents
from agents import compare_agents


class pickup():
    pass

AtLoctionA = ["Fox", "Chicken", "Feed"]
AtLoctionB = []
pickedUp = ""

def percept(self, agent):
    if ("Fox" in AtLoctionA and "Chicken" in AtLoctionA and "Feed" in AtLoctionA):
        self.pickup.AtLocationA[1]
        pickedUp = AtLoctionA[1]
        AtLoctionA.pop(1)
        #for demo purposes I'm using L as location A and R as location B
        agents.Direction.R
        AtLoctionB.append(pickedUp)

    else:
        agents.Direction.L
        self.pickup.AtloctionA[0]
        pickedUp = AtLoctionA[0]
        AtLoctionA.pop[0]

        agents.Direction.R
        AtLoctionB.append(pickedUp)
        
        self.pickup.AtloctionB[0]
        pickedUp = AtLoctionB[0]
        AtLoctionB.pop[0]
        agents.Direction.L
        AtLoctionA.append(pickedUp)

        self.pickup.AtloctionA[0]
        pickedUp = AtLoctionA[0]
        AtLoctionA.pop[0]

        agents.Direction.R
        AtLoctionB.append(pickedUp)

        agents.Direction.L 
        self.pickup.AtloctionA[0]
        pickedUp = AtLoctionA[0]
        AtLoctionA.pop[0]

        agents.Direction.R
        AtLoctionB.append(pickedUp)

# 6)

def TableDrivenAgentProgram(table):
    """
    [Figure 2.7]
    This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it, provide as table a dictionary of all
    {percept_sequence:action} pairs.
    """
    percepts = []

    def program(percept):
        percepts.append(percept)
        action = table.get(percepts)
        return action

    return program

#B. 

#    RandomVacuumAgent     TableDrivenVacuumAgent      ReflexVacuumAgent       ModelBasedVacuumAgent
#p    Cleaning Score         Cleaning score             Cleaning score          Cleaning score  
#e    Clean/Dirty            Clean/Dirty                Clean/Dirty             Clean/Dirty
#a    L/R, suck              L/R, suck                  L/R, suck               L/R, suck, noOp
#s    check for dirt         Table dirctions            check for dirt          check for dirt

#I have commented out this next section to allow me to selec the enviromnet for question C.

#environment = agents.TrivialVacuumEnvironment
#agents = [agents.RandomVacuumAgent, agents.TableDrivenVacuumAgent, agents.ReflexVacuumAgent, agents.ModelBasedVacuumAgent]
#result = compare_agents(environment, agents)

#performance_RandomVacuumAgent = result[1][1]
#performance_TableDrivenVacuumAgent = result[2][1]
#performance_ReflexVacuumAgent = result[3][1]
#performance_ModelBasedVacuumAgent = result[0][1]

# The TrivalVacuumEnvionment seems to show that the Random and reflex vacuuum agents seem to perform the best in this enviroment.
# The RandomVacuumAgent only cleans if there is dirt in the loction it is in. It can percive it all the loctions in the world are dirty and can then stop operating if so
# The TableDrivenVacuumAgent will follow a set of instuctions that are listed for the agent to follow. It seems to perform the worst which shows that it isn't effective at cleaning the environment efficently. As it seem to move as guided by the table instructions even if what the agent percives is different to its instuctions 
# The ReflexVacuumAgent performs well and returns a positive score in this environment. This is because the Reflex vaacum is able to percive if the loction it is at has dirt or not.

#C.

environment = agents.VacuumEnvironment
#agents = [agents.RandomVacuumAgent, agents.ReflexVacuumAgent, agents.ModelBasedVacuumAgent]
#result = compare_agents(environment, agents)

#performance_RandomVacuumAgent = result[1][1]
#performance_ReflexVacuumAgent = result[2][1]
#performance_ModelBasedVacuumAgent = result[0][1]

#print("Performance of RandomVacuumAgent:", performance_RandomVacuumAgent)
#print("Performance of ReflexVacuumAgent:", performance_ReflexVacuumAgent)
#print("Performance of ModelBasedVacuumAgent:", performance_ModelBasedVacuumAgent)

# The results of the agents show that the ModelBasedVacuumAgent performs best in this 1-D enviroment as it has a percept history of the world it is perceving. It considers the actions it takes allowing it to decide which move is best to take next as it learning about its environment allwoing it to improve overtime.

#For the other two agents it shows that they are far less efficent due to not having a percept history. This causes a more random decision approch to there movement in the 1-D enviroment.

#D.
#Goal:  Make sure the environment is clean

#Model of world
agents.ModelBasedVacuumAgent = {agents.loc_A: None, agents.loc_B: None}
agents = [agents.ModelBasedVacuumAgent]
result = compare_agents(environment, agents)

performance_RandomVacuumAgent = result[1][1]
performance_ReflexVacuumAgent = result[2][1]
performance_ModelBasedVacuumAgent = result[0][1]

print("Performance of RandomVacuumAgent:", performance_RandomVacuumAgent)
print("Performance of ReflexVacuumAgent:", performance_ReflexVacuumAgent)
print("Performance of ModelBasedVacuumAgent:", performance_ModelBasedVacuumAgent)