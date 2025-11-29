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
        
    # Do NOT force the final carry to 0.
    # If RHS has an extra leading digit (len(RHS) == max(len(LHS0), len(LHS1)) + 1),
    # then the final carry must be 1 and will equal that leading digit via the column constraint below.
    # Otherwise, the final carry will end up 0 naturally from the same constraint.
        
        # For each digit position, add constraints using only unary/binary by encoding pairs
        # Column equation: (x or 0) + (y or 0) + carry_in = result_digit + 10 * carry_out
        for i in range(n):
            pos_from_right = i

            # Indices into each word (right-aligned)
            l0_idx = len(LHS0) - 1 - pos_from_right
            l1_idx = len(LHS1) - 1 - pos_from_right
            lr_idx = len(RHS) - 1 - pos_from_right

            x = LHS0[l0_idx] if 0 <= l0_idx < len(LHS0) else None
            y = LHS1[l1_idx] if 0 <= l1_idx < len(LHS1) else None
            r = RHS[lr_idx]

            ci = f'C{pos_from_right}'
            co = f'C{pos_from_right + 1}'

            # Auxiliary variable PAIR_i encodes (x, y) as 10*x + y. Missing letters treated as 0.
            pair = f'PAIR{i}'
            problem.variables.append(pair)
            # Tight pair domain based on current letter domains (or 0 if missing)
            xd = problem.domains[x] if x else {0}
            yd = problem.domains[y] if y else {0}
            problem.domains[pair] = {10*vx + vy for vx in xd for vy in yd}

            # Link x to PAIR_i (tens digit) when x exists
            if x:
                def check_x_pair(vx, vp):
                    return (vp // 10) == vx
                problem.constraints.append(BinaryConstraint((x, pair), check_x_pair))

            # Link y to PAIR_i (ones digit) when y exists
            if y:
                def check_y_pair(vy, vp):
                    return (vp % 10) == vy
                problem.constraints.append(BinaryConstraint((y, pair), check_y_pair))

            # Auxiliary PSUM_i = (x or 0) + (y or 0)
            psum = f'PSUM{i}'
            problem.variables.append(psum)
            # Tight psum domain based on xd, yd
            psum_domain = {vx + vy for vx in xd for vy in yd}
            problem.domains[psum] = psum_domain

            def check_pair_psum(vp, vs):
                return ((vp // 10) + (vp % 10)) == vs
            problem.constraints.append(BinaryConstraint((pair, psum), check_pair_psum))

            # Combine PSUM and carry_in into COMB_i = 20*ci + psum
            comb = f'COMB{i}'
            problem.variables.append(comb)
            comb_domain = {20*ci_val + s for ci_val in {0,1} for s in psum_domain}
            problem.domains[comb] = comb_domain

            def check_ci_comb(vci, vc):
                return (vc // 20) == vci
            def check_psum_comb(vs, vc):
                return (vc % 20) == vs
            problem.constraints.append(BinaryConstraint((ci, comb), check_ci_comb))
            problem.constraints.append(BinaryConstraint((psum, comb), check_psum_comb))

            # SUM_i = psum + ci = (comb % 20) + (comb // 20)
            sum_i = f'SUM{i}'
            problem.variables.append(sum_i)
            sum_domain = {s + ci_val for ci_val in {0,1} for s in psum_domain}
            problem.domains[sum_i] = sum_domain

            def check_comb_sum(vc, vs):
                return ((vc % 20) + (vc // 20)) == vs
            problem.constraints.append(BinaryConstraint((comb, sum_i), check_comb_sum))

            # Link SUM to result digit and next carry: r == SUM % 10, co == SUM // 10
            def check_r_sum(vr, vs):
                return (vs % 10) == vr
            def check_co_sum(vco, vs):
                return (vs // 10) == vco
            problem.constraints.append(BinaryConstraint((r, sum_i), check_r_sum))
            problem.constraints.append(BinaryConstraint((co, sum_i), check_co_sum))
        
        return problem

    # Read a cryptarithmetic puzzle from a file
    @staticmethod
    def from_file(path: str) -> "CryptArithmeticProblem":
        with open(path, 'r') as f:
            return CryptArithmeticProblem.from_text(f.read())
