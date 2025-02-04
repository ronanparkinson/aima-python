#part A

from logic import Expr, tt_entails

# Define symbols for the logical expressions
H = Expr('H') # It is hot
AC = Expr('AC') # The air conditioner is on

# Define the premises using Expr
premise1 = Expr('==>', H, AC) # Premise 1: If it is hot, then the air conditioner is on
premise2 = H # Premise 2: It is hot

 # Combine the premises using logical AND
premises = premise1 & premise2
print(premises)

 # Define the conclusion
conclusion = AC # Conclusion: The air conditioner is on

# Check if the premises entail the conclusion
result = tt_entails(premises, conclusion)

# Output the result
print(f"Do the premises entail the conclusion? {result}")

 # Combine the premises using logical OR
premises = premise1 | premise2
print(premises)


 # Define the conclusion
conclusion = (H | AC) 

result = tt_entails(premises, conclusion)

print(f"Do the premises entail the conclusion? {result}")

 # Combine the premises using logical implies
premises = Expr('==>', premise1, premise2)
print(premises)

 # Define the conclusion
conclusion = (~ H) | AC

result = tt_entails(premises, conclusion)

print(f"Do the premises entail the conclusion? {result}")

#B.

