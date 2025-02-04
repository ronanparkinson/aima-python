import sys
sys.path.append('./aima-python')  # Ensure correct path to aima-python

from logic import Expr, PropKB

#1  Propositional Logic - Smart Home System
"""
Step 1: Represent the facts as propositional variables

We will represent the facts from the problem description using propositional variables:

    A: It's dark outside.
    B: The living room lights are on.
    C: It's cold outside.
    D: The heater is on.
    E: It's 7 pm.
    F: The temperature inside is below 18°C.

Step 2: Construct rules in propositional logic

The rules provided in the problem can be written in propositional logic as follows:

    If it's 7 pm, it is dark outside:
    E ⇒ A
    If it’s dark outside, the living room lights should be turned on:
    A ⇒ B
    If the temperature inside is below 18°C, it’s considered cold:
    F ⇒ C
    If the living room lights are on and it’s cold outside, the heater should be turned on:
    B ∧ C ⇒ D

Scenario 1: Bob enters his home at 7 pm, and the temperature outside is 17°C (below 18°C).

    Given facts: E=True,F=TrueE=True,F=True
    Inference steps:
        E⇒A: Since E=True, A=True (It’s dark outside).
        A⇒B: Since A=True, B=True (Living room lights are on).
        F⇒C: Since F=True, C=True (It’s cold outside).
        B∧C⇒D: Since both B=True and C=True, D=True (The heater is on).
    Result: The heater is on. (Expected: True)

Scenario 2: The temperature outside is 19°C (above 18°C).

    Given facts: E=True,F=False
    Inference steps:
        E⇒A: Since E=True, A=True (It’s dark outside).
        A⇒B: Since A=True, B=True (Living room lights are on).
        F⇒C: Since F=False, C=False (It’s not cold outside).
        B∧C⇒D: Since C=False, D=False (The heater is not on).
    Result: The heater is off. (Expected: False)

"""

# Step 1: Define the facts using propositional variables
A = Expr('A')  # It's dark outside
B = Expr('B')  # The living room lights are on
C = Expr('C')  # It's cold outside
D = Expr('D')  # The heater is on
E = Expr('E')  # It's 7 pm
F = Expr('F')  # The temperature inside is below 18°C

# Step 2: Construct rules in propositional logic
kb = PropKB()  # Initialize the knowledge base (KB)

# Add the rules to the KB
kb.tell(E |'==>'| A)           # E implies A (If it's 7 pm, it's dark outside)
kb.tell(A |'==>'|  B)           # A implies B (If it's dark outside, the living room lights are on)
kb.tell(F |'==>'|  C)           # F implies C (If the temperature is below 18°C, it's cold)
kb.tell(B & C |'==>'| D)       # B and C imply D (If the lights are on and it's cold, the heater is on)

# Step 3: Scenario 1 - Bob enters his home at 7 pm, and the temperature is below 18°C.
kb.tell(E)  # It's 7 pm (E is True)
kb.tell(F)  # The temperature is below 18°C (F is True)

# Perform inference to check if the heater (D) is on
print("Scenario 1: Is the heater on?", kb.ask_if_true(D))  # Expected: True

# Step 4: Scenario 2 - The temperature is above 18°C.
kb.retract(F)  # Now the temperature is not below 18°C (F is False)
kb.tell(~F)    # Assert that F is False (the temperature is not below 18°C)

# Perform inference to check if the heater (D) is on
print("Scenario 2: Is the heater on?", kb.ask_if_true(D))  # Expected: False

"""
#2 First-Order Logic - Smart Medical System

Step 1: Define the knowledge base in FOL

We will use predicates to define the facts:
    HasFever(x): x has a fever.
    HasCough(x): x has a cough.
    HasFlu(x): x has the flu.
    NeedsAntiviral(x): x needs antiviral medication.
    HasRash(x): x has a rash.
    HasItchiness(x): x has itchiness.
    HasAllergicReaction(x): x has an allergic reaction.
    NeedsAntihistamines(x): x needs antihistamines.

Step 2: Rules in FOL

We represent the rules from the problem description using FOL:

    If someone has a fever and a cough, they might have the flu:
    ∀x HasFever(x) ∧ HasCough(x) ⇒ HasFlu(x)
    
    If someone has the flu, they need antiviral medication:
    ∀x HasFlu(x) ⇒ NeedsAntiviral(x)
    
    If someone has a rash and itchiness, they might have an allergic reaction:
    ∀x HasRash(x) ∧ HasItchiness(x) ⇒ HasAllergicReaction(x)
    
    If someone has an allergic reaction, they need antihistamines:
    ∀x HasAllergicReaction(x) ⇒ NeedsAntihistamines(x)
    

Step 3: Given facts

    John has a fever and a cough:
    HasFever(John),HasCough(John)
    Alice has a rash:
    HasRash(Alice)

Step 4: Forward Chaining to Determine Treatments

Using forward chaining, we can infer:

    For John:
    Since John has a fever and a cough, 
      HasFever(John) ∧ HasCough(John) ⇒ HasFlu(John), so John has the flu.
        Since John has the flu, 
      HasFlu(John) ⇒ NeedsAntiviral(John), so John needs antiviral medication.

    For Alice:
        Since Alice has a rash, but no information about itchiness is provided, we cannot infer if Alice has an allergic reaction, and thus we cannot determine if she needs antihistamines.

Step 5: Results

    John's treatment: John needs antiviral medication.
    Alice's treatment: No treatment determined from the given information.
"""

from logic import FolKB, fol_fc_ask

# Step 1: Define the knowledge base in FOL
kb = FolKB()

# Step 2: Add rules in FOL

# If someone has a fever and a cough, they might have the flu
kb.tell(Expr('forall x (HasFever(x) & HasCough(x) ==> HasFlu(x))'))

# If someone has the flu, they need antiviral medication
kb.tell(Expr('forall x (HasFlu(x) ==> NeedsAntiviral(x))'))

# If someone has a rash and itchiness, they might have an allergic reaction
kb.tell(Expr('forall x (HasRash(x) & HasItchiness(x) ==> HasAllergicReaction(x))'))

# If someone has an allergic reaction, they need antihistamines
kb.tell(Expr('forall x (HasAllergicReaction(x) ==> NeedsAntihistamines(x))'))

# Step 3: Add facts about John and Alice

# John has a fever and a cough
kb.tell(Expr('HasFever(John)'))
kb.tell(Expr('HasCough(John)'))

# Alice has a rash (but no information about itchiness)
kb.tell(Expr('HasRash(Alice)'))

# Step 4: Use forward chaining to infer treatments

# For John, we expect him to need antiviral medication
query_john = fol_fc_ask(kb, Expr('NeedsAntiviral(John)'))
print("Does John need antiviral medication?", bool(query_john))  # Expected: True

# For Alice, no conclusion about antihistamines can be drawn (no info about itchiness)
query_alice = fol_fc_ask(kb, Expr('NeedsAntihistamines(Alice)'))
print("Does Alice need antihistamines?", bool(query_alice))  # Expected: False
