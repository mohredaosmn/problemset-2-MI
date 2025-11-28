from typing import Tuple
import re
from CSP import Assignment, Problem, UnaryConstraint, BinaryConstraint

#TODO (Optional): Import any builtin library or define any helper function you want to use

# This is a class to define for cryptarithmetic puzzles as CSPs
class CryptArithmeticProblem(Problem):
    LHS: Tuple[str, str]
    RHS: str

    # Convert an assignment into a string (so that is can be printed).
    def format_assignment(self, assignment: Assignment) -> str:
        LHS0, LHS1 = self.LHS
        RHS = self.RHS
        letters = set(LHS0 + LHS1 + RHS)
        formula = f"{LHS0} + {LHS1} = {RHS}"
        postfix = []
        valid_values = list(range(10))
        for letter in letters:
            value = assignment.get(letter)
            if value is None: continue
            if value not in valid_values:
                postfix.append(f"{letter}={value}")
            else:
                formula = formula.replace(letter, str(value))
        if postfix:
            formula = formula + " (" + ", ".join(postfix) +  ")" 
        return formula

    @staticmethod
    def from_text(text: str) -> 'CryptArithmeticProblem':
        # Given a text in the format "LHS0 + LHS1 = RHS", the following regex
        # matches and extracts LHS0, LHS1 & RHS
        # For example, it would parse "SEND + MORE = MONEY" and extract the
        # terms such that LHS0 = "SEND", LHS1 = "MORE" and RHS = "MONEY"
        pattern = r"\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*=\s*([a-zA-Z]+)\s*"
        match = re.match(pattern, text)
        if not match: raise Exception("Failed to parse:" + text)
        LHS0, LHS1, RHS = [match.group(i+1).upper() for i in range(3)]

        problem = CryptArithmeticProblem()
        problem.LHS = (LHS0, LHS1)
        problem.RHS = RHS

        # Extract all unique letters
        all_letters = set(LHS0 + LHS1 + RHS)
        
        # Create variables list - include all letters
        problem.variables = list(all_letters)
        
        # Create domains for letters (0-9)
        problem.domains = {letter: set(range(10)) for letter in all_letters}
        
        # Initialize constraints list
        problem.constraints = []
        
        # Unary constraints: leading letters cannot be 0
        if LHS0:
            first_letter_lhs0 = LHS0[0]
            problem.constraints.append(UnaryConstraint(first_letter_lhs0, lambda x: x != 0))
        if LHS1:
            first_letter_lhs1 = LHS1[0]
            problem.constraints.append(UnaryConstraint(first_letter_lhs1, lambda x: x != 0))
        if RHS:
            first_letter_rhs = RHS[0]
            problem.constraints.append(UnaryConstraint(first_letter_rhs, lambda x: x != 0))
        
        # Binary constraints: all letters must have different values
        letters_list = list(all_letters)
        for i in range(len(letters_list)):
            for j in range(i + 1, len(letters_list)):
                letter1, letter2 = letters_list[i], letters_list[j]
                problem.constraints.append(
                    BinaryConstraint((letter1, letter2), lambda x, y: x != y)
                )
        
        # Add auxiliary variables for carries
        # We need carries C0, C1, ..., C_n where n = len(RHS)
        # C0 is carry into rightmost digit (always 0)
        # Ci is carry out of digit i (0-indexed from right)
        n = len(RHS)
        for i in range(n + 1):
            carry_name = f'C{i}'
            problem.variables.append(carry_name)
            problem.domains[carry_name] = {0, 1}
        
        # C0 is always 0 (no initial carry)
        problem.constraints.append(UnaryConstraint('C0', lambda x: x == 0))
        
        # Cn (final carry) must be 0 (no overflow)
        final_carry = f'C{n}'
        problem.constraints.append(UnaryConstraint(final_carry, lambda x: x == 0))
        
        # For each digit position, add constraints
        # Column equation: digit0 + digit1 + carry_in = digit_result + 10 * carry_out
        for i in range(n):
            pos_from_right = i
            
            # Get the letter at this position (or None if out of range)
            l0_idx = len(LHS0) - 1 - pos_from_right
            l1_idx = len(LHS1) - 1 - pos_from_right
            lr_idx = len(RHS) - 1 - pos_from_right
            
            l0 = LHS0[l0_idx] if 0 <= l0_idx < len(LHS0) else None
            l1 = LHS1[l1_idx] if 0 <= l1_idx < len(LHS1) else None
            lr = RHS[lr_idx]  # RHS always has a digit at this position
            
            carry_in = f'C{pos_from_right}'
            carry_out = f'C{pos_from_right + 1}'
            
            # We need: (l0 or 0) + (l1 or 0) + carry_in = lr + 10 * carry_out
            # Since we can only use binary constraints, we create auxiliary variables
            
            # Add auxiliary variable for left side: sum_left = (l0 or 0) + (l1 or 0) + carry_in
            sum_var = f'SUM{pos_from_right}'
            problem.variables.append(sum_var)
            problem.domains[sum_var] = set(range(20))  # max 9+9+1=19
            
            # Now we need to link sum_var with the actual sum
            # We'll create binary constraints between sum_var and each input
            
            # If l0 exists, add constraint: sum_var - l0 must be achievable with l1 + carry_in
            if l0:
                # For each pair (l0_val, sum_val), check if sum_val - l0_val can be made by l1 + carry_in
                def make_l0_constraint(l0_letter=l0, l1_letter=l1):
                    def check(v0, vsum):
                        remainder = vsum - v0
                        if remainder < 0 or remainder > 19:
                            return False
                        # Check if remainder can be formed by (l1 or 0) + carry_in
                        for vci in [0, 1]:
                            if l1_letter:
                                # Need to check if any l1 value works
                                for v1 in range(10):
                                    if v1 + vci == remainder:
                                        return True
                            else:
                                # l1 is 0
                                if vci == remainder:
                                    return True
                        return False
                    return check
                problem.constraints.append(BinaryConstraint((l0, sum_var), make_l0_constraint()))
            
            # If l1 exists, add similar constraint
            if l1:
                def make_l1_constraint(l0_letter=l0, l1_letter=l1):
                    def check(v1, vsum):
                        remainder = vsum - v1
                        if remainder < 0 or remainder > 19:
                            return False
                        for vci in [0, 1]:
                            if l0_letter:
                                for v0 in range(10):
                                    if v0 + vci == remainder:
                                        return True
                            else:
                                if vci == remainder:
                                    return True
                        return False
                    return check
                problem.constraints.append(BinaryConstraint((l1, sum_var), make_l1_constraint()))
            
            # Add constraint for carry_in
            def make_ci_constraint(l0_letter=l0, l1_letter=l1):
                def check(vci, vsum):
                    remainder = vsum - vci
                    if remainder < 0 or remainder > 18:
                        return False
                    # Check if remainder can be formed by (l0 or 0) + (l1 or 0)
                    for v0 in ([0] if not l0_letter else range(10)):
                        for v1 in ([0] if not l1_letter else range(10)):
                            if v0 + v1 == remainder:
                                return True
                    return False
                return check
            problem.constraints.append(BinaryConstraint((carry_in, sum_var), make_ci_constraint()))
            
            # Now link sum_var to lr and carry_out: sum_var = lr + 10 * carry_out
            
            # Constraint between lr and sum_var
            def check_lr_sum(vlr, vsum):
                # Check if there's a carry_out value (0 or 1) such that vlr + 10*carry_out = vsum
                return (vsum == vlr) or (vsum == vlr + 10)
            problem.constraints.append(BinaryConstraint((lr, sum_var), check_lr_sum))
            
            # Constraint between carry_out and sum_var
            def check_co_sum(vco, vsum):
                # Check if there's an lr value (0-9) such that lr + 10*vco = vsum
                for vlr in range(10):
                    if vlr + 10 * vco == vsum:
                        return True
                return False
            problem.constraints.append(BinaryConstraint((carry_out, sum_var), check_co_sum))
        
        return problem

    # Read a cryptarithmetic puzzle from a file
    @staticmethod
    def from_file(path: str) -> "CryptArithmeticProblem":
        with open(path, 'r') as f:
            return CryptArithmeticProblem.from_text(f.read())
