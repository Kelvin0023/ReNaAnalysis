
def count_standard_error(true_label, pred_label):
    count = 0
    for elem1, elem2 in zip(true_label, pred_label):
        if elem1 == 0 and elem2 == 1:
            count += 1
    return count

def count_target_error(true_label, pred_label):
    count = 0
    for elem1, elem2 in zip(true_label, pred_label):
        if elem1 == 1 and elem2 == 0:
            count += 1
    return count