from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented

# This function applies 1-Consistency to the problem.
# In other words, it modifies the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints are removed from the problem (they are no longer needed).
# The function returns False if any domain becomes empty. Otherwise, it returns True.
def one_consistency(problem: Problem) -> bool:
    remaining_constraints = []
    solvable = True
    for constraint in problem.constraints:
        if not isinstance(constraint, UnaryConstraint):
            remaining_constraints.append(constraint)
            continue
        variable = constraint.variable
        new_domain = {value for value in problem.domains[variable] if constraint.condition(value)}
        if not new_domain:
            solvable = False
        problem.domains[variable] = new_domain
    problem.constraints = remaining_constraints
    return solvable

# This function returns the variable that should be picked based on the MRV heuristic.
# NOTE: We don't use the domains inside the problem, we use the ones given by the "domains" argument 
#       since they contain the current domains of unassigned variables only.
# NOTE: If multiple variables have the same priority given the MRV heuristic, 
#       we order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    _, _, variable = min((len(domains[variable]), index, variable) for index, variable in enumerate(problem.variables) if variable in domains)
    return variable

# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    # Forward checking: After assigning a value to a variable, we update the domains of unassigned neighbors
    # by removing values that are inconsistent with the assignment based on binary constraints.
    
    # Iterate through all constraints in the problem
    for constraint in problem.constraints:
        # We only care about binary constraints involving the assigned variable
        if not isinstance(constraint, BinaryConstraint):
            continue
        
        # Check if this constraint involves the assigned variable
        if assigned_variable not in constraint.variables:
            continue
        
        # Get the other variable involved in this constraint
        other_variable = constraint.get_other(assigned_variable)
        
        # If the other variable is already assigned (not in domains), skip it
        if other_variable not in domains:
            continue
        
        # Create a new domain for the other variable by filtering values
        # that satisfy the constraint with the assigned value
        new_domain = set()
        for value in domains[other_variable]:
            # Check if the constraint is satisfied with this value combination
            # We need to determine which variable is first in the constraint tuple
            if constraint.variables[0] == assigned_variable:
                # assigned_variable is first, other_variable is second
                if constraint.condition(assigned_value, value):
                    new_domain.add(value)
            else:
                # other_variable is first, assigned_variable is second
                if constraint.condition(value, assigned_value):
                    new_domain.add(value)
        
        # If the domain becomes empty, this assignment is inconsistent
        if not new_domain:
            return False
        
        # Update the domain of the other variable
        domains[other_variable] = new_domain
    
    # All domains are non-empty, so the assignment is potentially consistent
    return True

# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    # This heuristic orders values by how much they constrain the neighbors' domains.
    # Values that eliminate fewer options from neighbors' domains are preferred (least restraining).
    
    # Dictionary to store how many values each value of variable_to_assign eliminates from neighbors
    value_restraint_count = {}
    
    # For each possible value in the domain of the variable to assign
    for value in domains[variable_to_assign]:
        restraint_count = 0  # Count how many neighbor values this eliminates
        
        # Check all binary constraints involving this variable
        for constraint in problem.constraints:
            if not isinstance(constraint, BinaryConstraint):
                continue
            
            # Check if this constraint involves the variable we're assigning
            if variable_to_assign not in constraint.variables:
                continue
            
            # Get the other variable in the constraint
            other_variable = constraint.get_other(variable_to_assign)
            
            # If the other variable is already assigned, skip it
            if other_variable not in domains:
                continue
            
            # Count how many values in the other variable's domain would be eliminated
            # by assigning this value to variable_to_assign
            for other_value in domains[other_variable]:
                # Check if this combination satisfies the constraint
                # If it doesn't satisfy, it means we're eliminating this value
                if constraint.variables[0] == variable_to_assign:
                    # variable_to_assign is first, other_variable is second
                    if not constraint.condition(value, other_value):
                        restraint_count += 1
                else:
                    # other_variable is first, variable_to_assign is second
                    if not constraint.condition(other_value, value):
                        restraint_count += 1
        
        value_restraint_count[value] = restraint_count
    
    # Sort values by restraint count (ascending), then by value itself (ascending) for ties
    # This ensures least restraining values come first, and stable ordering for equal restraint
    sorted_values = sorted(domains[variable_to_assign], 
                          key=lambda v: (value_restraint_count[v], v))
    
    return sorted_values

# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:
    # Backtracking search with forward checking, MRV heuristic, and least restraining value heuristic
    
    # Step 1: Apply 1-consistency to enforce unary constraints
    # This removes values from domains that don't satisfy unary constraints
    if not one_consistency(problem):
        # If 1-consistency makes the problem unsolvable, return None immediately
        # Don't call problem.is_complete as per requirements
        return None
    
    # Step 2: Initialize the assignment and domains for unassigned variables
    assignment = {}  # Empty assignment to start
    
    # Create initial domains dictionary with only unassigned variables
    domains = {var: problem.domains[var].copy() for var in problem.variables}
    
    # Helper function for recursive backtracking
    def backtrack(assignment: Assignment, domains: Dict[str, set]) -> Optional[Assignment]:
        # Check if the assignment is complete (base case)
        if problem.is_complete(assignment):
            # Verify that it satisfies all constraints
            if problem.satisfies_constraints(assignment):
                return assignment
            return None
        
        # Step 3: Select the next variable to assign using MRV heuristic
        # Only consider unassigned variables (those still in domains)
        unassigned_domains = {var: dom for var, dom in domains.items() if var not in assignment}
        
        if not unassigned_domains:
            # All variables are assigned but is_complete returned False
            # This shouldn't happen in a well-formed problem
            return None
        
        variable = minimum_remaining_values(problem, unassigned_domains)
        
        # Step 4: Order the values using least restraining value heuristic
        ordered_values = least_restraining_values(problem, variable, unassigned_domains)
        
        # Step 5: Try each value in order
        for value in ordered_values:
            # Create a new assignment with this value
            new_assignment = assignment.copy()
            new_assignment[variable] = value
            
            # Create a copy of domains for forward checking
            new_domains = {var: domains[var].copy() for var in domains if var != variable}
            
            # Step 6: Apply forward checking
            # This updates new_domains based on the new assignment
            if forward_checking(problem, variable, value, new_domains):
                # Forward checking succeeded, continue with recursion
                result = backtrack(new_assignment, new_domains)
                if result is not None:
                    return result
            # If forward checking fails, this branch is pruned
            # We don't call problem.is_complete on pruned assignments
        
        # No value worked, backtrack
        return None
    
    # Start the backtracking search with initial empty assignment
    # The is_complete check will happen inside backtrack
    return backtrack(assignment, domains)