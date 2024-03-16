import os
import argparse
import jsonlines

class Task:
    HUMANITARIAN = 'humanitarian'
    INFORMATIVE = 'informative'
    DAMAGE = 'damage'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='C:\\Users\\Robert_Popovici\\Desktop\\licenta\\datasets\\humanitarian_orig')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--task', type=str, default='humanitarian')
    return parser.parse_args()

def get_labels(task):
    if task == Task.HUMANITARIAN:
        return ['not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort', 'infrastructure_and_utility_damage', 'affected_individuals']
    elif task == Task.INFORMATIVE:
        return ['informative', 'uninformative']
    elif task == Task.DAMAGE:
        return ['damage', 'no damage']
    else:
        raise Exception(f'Task {task} is not supported')


def main():
    args = get_arguments()

    filepath = os.path.join(args.dir, args.file)
    dataset = [row for row in jsonlines.open(filepath)]

    labels = get_labels(args.task)
    counter = {k: 0 for k in labels}

    for row in dataset:
        label = row['label']
        if label not in counter:
            raise Exception(f'Label {label} is not supported')
        counter[label] += 1

    for k, v in counter.items():
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()
