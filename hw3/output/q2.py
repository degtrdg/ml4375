init_states = [10,0,0,-5]

def s1(states):
    return 10 + 0.5 * max(states[0], states[1])

def s2(states):
    return 0 + 0.5 * max(states[1], states[2])

def s3(states):
    return 0 + 0.5 * max(0.5*states[0] + 0.5*states[2], states[3])

def s4(states):
    return -5 + 0.5 * max(states[0], states[3])

def next_states(states):
    return [s1(states), s2(states), s3(states), s4(states)]

def iterate_states(states, epsilon, count):
    print(str(count) + ": ", end="")
    print(states)
    next = next_states(states)
    if not all([abs(z[0] - z[1]) < epsilon for z in zip(next, states)]):
        iterate_states(next, epsilon, count+1)

iterate_states(init_states, 0.0001, 0)
