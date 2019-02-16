def argmin(dictionary):
    if not dictionary: return None
    min_val = min(dictionary.values())
    return [k for k in dictionary if dictionary[k] == min_val][0]

def argmax(dictionary):
    if not dictionary: return None
    max_val = max(dictionary.values())
    return [k for k in dictionary if dictionary[k] == max_val][0]

