import json
import sys
import argparse
import logging
import pls_sklearn
import functions
import cPickle as pickle

logging.basicConfig(level=logging.DEBUG)
config = json.loads(open(sys.argv[1]).read())
logging.info("START")

pipeline = functions.load_pipeline(config)

predictors = functions.open_uris(config['predictors'])
logging.info("{} predictors".format(len(predictors)))

predictands = functions.open_uris(config['predictands'])
logging.info("{} predictands".format(len(predictands)))

locations = functions.features_to_centroids(predictands[0]['variables'][0]['field'].features())
logging.info("{} locations found".format(len(locations)))


#locations = locations[1400:1403]

print pipeline

results = {}
for component in pipeline:
	name = component['name']
	print name

	results[name] = {}

	for location in locations:

		try:
			args = [eval(input) for input in component['inputs']]
		except:
			args = [results[input][location.wkb.encode('hex')] for input in component['inputs']]

		output = component['function'](*args, **component['params'])

		locstring = location.wkb.encode('hex')		
		results[name][locstring] = component['function'](*args, **component['params'])
		
		print type(results[name][location.wkb.encode('hex')])


#for loc in results['PLS_transform']:
#	print loc
