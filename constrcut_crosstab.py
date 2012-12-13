import os
import gzip
import csv
from collections import Counter, defaultdict
import argparse
#
import numpy
import pandas


parser = argparse.ArgumentParser()
parser.add_argument('top_n_charities', type=int)
parser.add_argument('top_n_persons', type=int)
args = parser.parse_args()
top_n_charities = args.top_n_charities
top_n_persons = args.top_n_persons

data_dir = '/usr/local/ssci.data/CharityNet/CharityNet_Normalized_18Sept2012'
transaction_file = 'transactions.csv.gz'
transaction_full_file = os.path.join(data_dir, transaction_file)
desired_types = dict(person_id=int, charity_id=int, amount=float)

transactions_dict = None
with gzip.open(transaction_full_file) as fh:
    csv_reader = csv.reader(fh)
    header = csv_reader.next()
    transactions_dict = dict(zip(desired_types.keys(), [[] for el in desired_types]))
    for row in csv_reader:
        row_tuples = zip(header, row)
        for column, value in row_tuples:
            if column not in desired_types: continue
            which_type = desired_types[column]
            transactions_dict[column].append(which_type(value))

def get_most_common(in_list, top_n):
    top_tuples = Counter(in_list).most_common(top_n)
    top_set = set([el[0] for el in top_tuples])
    return top_set

transactions_F = pandas.DataFrame(transactions_dict)

top_charities = get_most_common(transactions_F['charity_id'], top_n_charities)
top_persons = get_most_common(transactions_F['person_id'], top_n_persons)
is_top_charity = lambda el: el in top_charities
is_top_person = lambda el: el in top_persons
is_top_charities = transactions_F['charity_id'].map(is_top_charity)
is_top_persons = transactions_F['person_id'].map(is_top_person)

top_F = transactions_F[is_top_charities & is_top_persons]
frame = pandas.DataFrame(index=set(top_F['person_id']),
                         columns=set(top_F['charity_id']))

for person, charity, amount in \
        zip(top_F['person_id'], top_F['charity_id'], top_F['amount']):
    frame[charity][person] = amount

frame = frame.dropna(how='all', axis=1)
frame[numpy.isnan(frame)] = 0
frame = frame.fillna(0)
row_sum = frame.sum(axis=1)
for column in frame:
    frame[column] /= row_sum

filename = 'top_' + str(top_n_charities) + '_by_' + str(top_n_persons) + '.csv'
frame.to_csv(filename, header=False, index=False)
