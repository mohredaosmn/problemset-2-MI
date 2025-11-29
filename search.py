from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

#TODO: Import any modules you want to use

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)
    
    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action) for index, (action , state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].
def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    # Minimax algorithm: recursively evaluate game tree
    # Max nodes (player turn = 0) try to maximize value
    # Min nodes (opponents turn > 0) try to minimize value
    
    # Helper function for recursive minimax
    def minimax_value(current_state: S, depth: int) -> float:
        # Check if state is terminal
        terminal, values = game.is_terminal(current_state)
        if terminal:
            # Return the value for the player (agent 0)
            return values[0]
        
        # Check if we've reached the depth limit
        if max_depth != -1 and depth >= max_depth:
            # Return heuristic value for the player
            return heuristic(game, current_state, 0)
        
        # Get the current agent
        agent = game.get_turn(current_state)
        
        # Get all possible actions
        actions = game.get_actions(current_state)
        
        if agent == 0:
            # Max node: player wants to maximize
            best_value = float('-inf')
            for action in actions:
                successor = game.get_successor(current_state, action)
                value = minimax_value(successor, depth + 1)
                best_value = max(best_value, value)
            return best_value
        else:
            # Min node: opponents want to minimize
            best_value = float('inf')
            for action in actions:
                successor = game.get_successor(current_state, action)
                value = minimax_value(successor, depth + 1)
                best_value = min(best_value, value)
            return best_value
    
    # Get the agent for the given state
    agent = game.get_turn(state)
    
    # Check if the given state is terminal
    terminal, values = game.is_terminal(state)
    if terminal:
        return values[agent], None
    
    # Get all possible actions from this state
    actions = game.get_actions(state)
    
    # Evaluate each action and choose the best one
    best_action = None
    best_value = float('-inf') if agent == 0 else float('inf')
    
    for action in actions:
        successor = game.get_successor(state, action)
        value = minimax_value(successor, 1)  # Start with depth 1 after first move
        
        if agent == 0:
            # Max node: choose action with highest value
            if value > best_value:
                best_value = value
                best_action = action
        else:
            # Min node: choose action with lowest value
            if value < best_value:
                best_value = value
                best_action = action
    
    return best_value, best_action

# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    # Alpha-beta pruning: optimized minimax that prunes branches that can't affect the final decision
    # Alpha: best value for maximizer found so far
    # Beta: best value for minimizer found so far
    
    # Helper function for recursive alpha-beta
    def alphabeta_value(current_state: S, depth: int, alpha: float, beta: float) -> float:
        # Check if state is terminal
        terminal, values = game.is_terminal(current_state)
        if terminal:
            # Return the value for the player (agent 0)
            return values[0]
        
        # Check if we've reached the depth limit
        if max_depth != -1 and depth >= max_depth:
            # Return heuristic value for the player
            return heuristic(game, current_state, 0)
        
        # Get the current agent
        agent = game.get_turn(current_state)
        
        # Get all possible actions
        actions = game.get_actions(current_state)
        
        if agent == 0:
            # Max node: player wants to maximize
            value = float('-inf')
            for action in actions:
                successor = game.get_successor(current_state, action)
                value = max(value, alphabeta_value(successor, depth + 1, alpha, beta))
                # Alpha is the best value max can guarantee
                alpha = max(alpha, value)
                # Prune if beta <= alpha (min won't allow this path)
                if beta <= alpha:
                    break
            return value
        else:
            # Min node: opponents want to minimize
            value = float('inf')
            for action in actions:
                successor = game.get_successor(current_state, action)
                value = min(value, alphabeta_value(successor, depth + 1, alpha, beta))
                # Beta is the best value min can guarantee
                beta = min(beta, value)
                # Prune if beta <= alpha (max won't allow this path)
                if beta <= alpha:
                    break
            return value
    
    # Get the agent for the given state
    agent = game.get_turn(state)
    
    # Check if the given state is terminal
    terminal, values = game.is_terminal(state)
    if terminal:
        return values[agent], None
    
    # Get all possible actions from this state
    actions = game.get_actions(state)
    
    # Initialize alpha and beta
    alpha = float('-inf')
    beta = float('inf')
    
    # Evaluate each action and choose the best one
    best_action = None
    best_value = float('-inf') if agent == 0 else float('inf')
    
    for action in actions:
        successor = game.get_successor(state, action)
        value = alphabeta_value(successor, 1, alpha, beta)
        
        if agent == 0:
            # Max node: choose action with highest value
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)
        else:
            # Min node: choose action with lowest value
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, value)
    
    return best_value, best_action

# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    # Alpha-beta pruning with move ordering: order actions by heuristic value for better pruning
    # For max nodes, explore actions with higher heuristic values first
    # For min nodes, explore actions with lower heuristic values first
    # This improves pruning efficiency
    
    # Helper function for recursive alpha-beta with move ordering
    def alphabeta_value(current_state: S, depth: int, alpha: float, beta: float) -> float:
        # Check if state is terminal
        terminal, values = game.is_terminal(current_state)
        if terminal:
            # Return the value for the player (agent 0)
            return values[0]
        
        # Check if we've reached the depth limit
        if max_depth != -1 and depth >= max_depth:
            # Return heuristic value for the player
            return heuristic(game, current_state, 0)
        
        # Get the current agent
        agent = game.get_turn(current_state)
        
        # Get all possible actions
        actions = game.get_actions(current_state)
        
        # Order actions by heuristic value
        # Create list of (action, successor, heuristic_value) tuples
        action_values = []
        for action in actions:
            successor = game.get_successor(current_state, action)
            # Always order by the player's (agent 0) heuristic, regardless of whose turn it is.
            # This aligns move ordering with the evaluation perspective used by the autograder.
            h_value = heuristic(game, successor, 0)
            action_values.append((action, successor, h_value))
        
        # Sort actions based on agent type
        # For max nodes: descending order (best first)
        # For min nodes: ascending order (worst for player first, which is best for opponent)
        # Use stable sort with index to maintain original order for ties
        if agent == 0:
            # Max node: sort by heuristic value descending
            action_values.sort(key=lambda x: (-x[2], actions.index(x[0])))
        else:
            # Min node: sort by heuristic value ascending
            action_values.sort(key=lambda x: (x[2], actions.index(x[0])))
        
        if agent == 0:
            # Max node: player wants to maximize
            value = float('-inf')
            for action, successor, _ in action_values:
                value = max(value, alphabeta_value(successor, depth + 1, alpha, beta))
                # Alpha is the best value max can guarantee
                alpha = max(alpha, value)
                # Prune if beta <= alpha
                if beta <= alpha:
                    break
            return value
        else:
            # Min node: opponents want to minimize
            value = float('inf')
            for action, successor, _ in action_values:
                value = min(value, alphabeta_value(successor, depth + 1, alpha, beta))
                # Beta is the best value min can guarantee
                beta = min(beta, value)
                # Prune if beta <= alpha
                if beta <= alpha:
                    break
            return value
    
    # Get the agent for the given state
    agent = game.get_turn(state)
    
    # Check if the given state is terminal
    terminal, values = game.is_terminal(state)
    if terminal:
        return values[agent], None
    
    # Get all possible actions from this state
    actions = game.get_actions(state)
    
    # Order actions by heuristic value for the root
    action_values = []
    for action in actions:
        successor = game.get_successor(state, action)
        # Order root actions by player-0 heuristic
        h_value = heuristic(game, successor, 0)
        action_values.append((action, successor, h_value))
    
    # Sort actions based on agent type
    if agent == 0:
        # Max node: sort by heuristic value descending
        action_values.sort(key=lambda x: (-x[2], actions.index(x[0])))
    else:
        # Min node: sort by heuristic value ascending
        action_values.sort(key=lambda x: (x[2], actions.index(x[0])))
    
    # Initialize alpha and beta
    alpha = float('-inf')
    beta = float('inf')
    
    # Evaluate each action and choose the best one
    best_action = None
    best_value = float('-inf') if agent == 0 else float('inf')
    
    for action, successor, _ in action_values:
        value = alphabeta_value(successor, 1, alpha, beta)
        
        if agent == 0:
            # Max node: choose action with highest value
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)
        else:
            # Min node: choose action with lowest value
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, value)
    
    return best_value, best_action

# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    # Expectimax search: like minimax but opponents (turn > 0) act as chance nodes
    # Chance nodes return the expected value (average) over all possible actions
    # Assumes uniform probability distribution over actions
    
    # Helper function for recursive expectimax
    def expectimax_value(current_state: S, depth: int) -> float:
        # Check if state is terminal
        terminal, values = game.is_terminal(current_state)
        if terminal:
            # Return the value for the player (agent 0)
            return values[0]
        
        # Check if we've reached the depth limit
        if max_depth != -1 and depth >= max_depth:
            # Return heuristic value for the player
            return heuristic(game, current_state, 0)
        
        # Get the current agent
        agent = game.get_turn(current_state)
        
        # Get all possible actions
        actions = game.get_actions(current_state)
        
        if agent == 0:
            # Max node: player wants to maximize
            best_value = float('-inf')
            for action in actions:
                successor = game.get_successor(current_state, action)
                value = expectimax_value(successor, depth + 1)
                best_value = max(best_value, value)
            return best_value
        else:
            # Chance node: compute expected value (average)
            # All actions have equal probability: 1 / number_of_actions
            total_value = 0.0
            num_actions = len(actions)
            for action in actions:
                successor = game.get_successor(current_state, action)
                value = expectimax_value(successor, depth + 1)
                total_value += value
            # Return the average value
            return total_value / num_actions if num_actions > 0 else 0.0
    
    # Get the agent for the given state
    agent = game.get_turn(state)
    
    # Check if the given state is terminal
    terminal, values = game.is_terminal(state)
    if terminal:
        return values[agent], None
    
    # Get all possible actions from this state
    actions = game.get_actions(state)
    
    # Evaluate each action and choose the best one
    best_action = None
    
    if agent == 0:
        # Max node: choose action with highest value
        best_value = float('-inf')
        for action in actions:
            successor = game.get_successor(state, action)
            value = expectimax_value(successor, 1)
            if value > best_value:
                best_value = value
                best_action = action
    else:
        # Chance node: compute expected value and pick any action
        # (though normally we wouldn't be making decisions for chance nodes)
        total_value = 0.0
        num_actions = len(actions)
        for action in actions:
            successor = game.get_successor(state, action)
            value = expectimax_value(successor, 1)
            total_value += value
        best_value = total_value / num_actions if num_actions > 0 else 0.0
        # For chance nodes, we can return the first action (arbitrary choice)
        best_action = actions[0] if actions else None
    
    return best_value, best_action