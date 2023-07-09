import application.pipe as pp
import json
import sys
import argparse

def main_test_muheqa():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="HOTPOT", type=str, choices=['HOTPOT', 'LC-QuAD'],\
                        help="Which dataset use to evaluate the model.")
	parser.add_argument("--decom_type", default="bridge", type=str, choices=['bridge', 'intersection'],\
                        help="Which decomposition method use.")
	args = parser.parse_args()
	
	# PATH of the decomposed questions
	path = '/PATH'

	print('Loading single questions...')

	with open(path, 'r') as f:
		lines = json.load(f)
	f.close()

	count = 0
	count_error = 0
	pipe = pp.Pipe()

	answers = {}
	answers2 = {}

	print('Answering single questions...')
	exc_list = ["Error","http","Q0","Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9"]
	for line in lines:
		sys.stdout.flush()
		for item in lines[line][0:2]:
			# Get the answers of the first questions
			answers_q1 = pipe.get_responses(item[0],max_answers=2,wikipedia=True,dbpedia=False,d4c=False)
			answers_q1_dict = []
			for ansq1 in answers_q1:
				answers_q1_dict.append({"text": str(ansq1["answer"]), "probability":ansq1["confidence"], "logit": 1.0, "no_answer_logit": 1.0,"evidence":ansq1["evidence"],"question_q1":item[0]})
			if not answers_q1_dict:
				answers_q1_dict.append({"text": "Error", "probability":0.0, "logit": 1.0, "no_answer_logit": 1.0,"evidence":"Error","question_q1":item[0]})
			answers[line] = answers_q1_dict

			if args.decom_type == "bridge":
				sub_count = 0

				for ans in answers[line]:
					# Replace part of the seconds questions with answers of the first question
					new_q = item[1].replace("[ANSWER]", ans['text'])
					answers_q2_dict = []

					if any(substr in ans['text'] for substr in exc_list):
						count_error += 1
						for i in range(0,2):
							answers_q2_dict.append({"text": "Error", "probability":0.0, "logit": 1.0, "no_answer_logit": 1.0,"evidence":"Error","multi_question":item[2],"question_q1":item[0],"question_q2":new_q})
					else:	
						# Get the answers of the seconds questions in bridge mode
						answers_q2 = pipe.get_responses(new_q,max_answers=2,wikipedia=True,dbpedia=False,d4c=False)
						for ansq2 in answers_q2:
							if any(substr in str(ansq2['answer']) for substr in exc_list):
								answers_q2_dict.append({"text": "Error", "probability":0.0, "logit": 1.0, "no_answer_logit": 1.0,"evidence":"Error","multi_question":item[2],"question_q1":item[0],"question_q2":new_q})
							else:
								answers_q2_dict.append({"text": str(ansq2["answer"]), "probability":ansq2["confidence"], "logit": 1.0, "no_answer_logit": 1.0,"evidence":ansq2["evidence"],"multi_question":item[2],"question_q1":item[0],"question_q2":new_q})
					answers2[line+"-"+str(sub_count)] = answers_q2_dict
					sub_count += 1
			else:
				# Get the answers of the seconds questions in intersection mode
				answers_q2 = pipe.get_responses(item[1],max_answers=2,wikipedia=True,dbpedia=False,d4c=False)
				answers_q2_dict = []
				for ansq2 in answers_q2:
					if any(substr in str(ansq2['answer']) for substr in exc_list):
						answers_q2_dict.append({"text": "Error", "probability":0.0, "logit": 1.0, "no_answer_logit": 1.0,"evidence":"Error","multi_question":item[2],"question_q1":item[0],"question_q2":item[1]})
					else:
						answers_q2_dict.append({"text": str(ansq2["answer"]), "probability":ansq2["confidence"], "logit": 1.0, "no_answer_logit": 1.0,"evidence":ansq2["evidence"],"multi_question":item[2],"question_q1":item[0],"question_q2":item[1]})
				answers2[line] = answers_q2_dict
			

			count += 1

	print(f'There are {count_error} answers for the first simple question that did not find a valid answer')

	with open(f'{args.dataset}_sample_dev_{args.decom_type}_1_nbest_predictions.json', 'w') as f:
		json.dump(answers, f)
	f.close()

	with open(f'{args.dataset}_sample_dev_{args.decom_type}_2_nbest_predictions.json', 'w') as f:
		json.dump(answers2, f)
	f.close()

	print('Saved single answers')

	sys.stdout.flush()

if __name__ == '__main__':
    main_test_muheqa()
