#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
public:
	const static int max_sentence_length = 1024;

public:
	// node instances
	

	
	vector<LookupNode> _char_inputs; // 
	vector<LookupNode> _char_inputs_r; // random initial
	vector<ConcatNode> _char_concat;

	vector<LookupNode> _l_bichar_inputs; //
	vector<LookupNode> _l_bichar_inputs_r; //
	vector<ConcatNode> _l_bichar_concat;

	vector<LookupNode> _r_bichar_inputs; //
	vector<LookupNode> _r_bichar_inputs_r; //
	vector<ConcatNode> _r_bichar_concat;

	vector<ConcatNode> _l_concat_inputs;	//char+
	vector<ConcatNode> _r_concat_inputs;

	vector<UniNode> _NonLinear_l;
	vector<UniNode> _NonLinear_r;

	LSTM1Builder _lstm_left;
	LSTM1Builder _lstm_right;

	vector<ConcatNode> _lstm_concat;

	vector<UniNode> _character;
	vector<LinearNode> _neural_outputs;

	Graph *_pcg;

public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_char_inputs.resize(sent_length);
		_l_bichar_inputs.resize(sent_length);
		_r_bichar_inputs.resize(sent_length);
		
		_char_inputs_r.resize(sent_length);
		_l_bichar_inputs_r.resize(sent_length);
		_r_bichar_inputs_r.resize(sent_length);

		_char_concat.resize(sent_length);
		_l_bichar_concat.resize(sent_length);
		_r_bichar_concat.resize(sent_length);

		_l_concat_inputs.resize(sent_length);
		_r_concat_inputs.resize(sent_length);
	
		_NonLinear_l.resize(sent_length);
		_NonLinear_r.resize(sent_length);

		_lstm_left.resize(sent_length);
		_lstm_right.resize(sent_length);
		_lstm_concat.resize(sent_length);
		_character.resize(sent_length);
		_neural_outputs.resize(sent_length);
	}

	inline void clear(){
		_char_inputs.clear();
		_l_bichar_inputs.clear();
		_r_bichar_inputs.clear();
		
		_char_inputs_r.clear();
		_l_bichar_inputs_r.clear();
		_r_bichar_inputs_r.clear();

		_char_concat.clear();
		_l_bichar_concat.clear();
		_r_bichar_concat.clear();

		_l_concat_inputs.clear();
		_r_concat_inputs.clear();
		
		_NonLinear_l.clear();
		_NonLinear_r.clear();
		
		_lstm_left.clear();
		_lstm_right.clear();
		_lstm_concat.clear();
		_character.clear();
		_neural_outputs.clear();
	}

public:
	inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts){
		_pcg = pcg;
		for (int idx = 0; idx < _char_inputs.size(); idx++) {
			_char_inputs[idx].setParam(&model.chars);
			_char_inputs[idx].init(opts.charDim, opts.dropProb);
			_l_bichar_inputs[idx].setParam(&model.l_bichars);
			_l_bichar_inputs[idx].init(opts.bicharDim, opts.dropProb);

			_r_bichar_inputs[idx].setParam(&model.r_bichars);
			_r_bichar_inputs[idx].init(opts.bicharDim, opts.dropProb);


			_char_inputs_r[idx].setParam(&model.chars_r);
			_char_inputs_r[idx].init(opts.characterSize, opts.dropProb);
			_l_bichar_inputs_r[idx].setParam(&model.l_bichars_r);
			_l_bichar_inputs_r[idx].init(opts.characterSize, opts.dropProb);

			_r_bichar_inputs_r[idx].setParam(&model.r_bichars_r);
			_r_bichar_inputs_r[idx].init(opts.characterSize, opts.dropProb);

			
			_char_concat[idx].init(opts.charDim + opts.characterSize, -1);
			_l_bichar_concat[idx].init(opts.bicharDim + opts.characterSize, -1);
			_r_bichar_concat[idx].init(opts.bicharDim + opts.characterSize, -1);

			_l_concat_inputs[idx].init(opts.charDim + opts.bicharDim + opts.characterSize * 2, -1);
			_r_concat_inputs[idx].init(opts.charDim + opts.bicharDim + opts.characterSize * 2, -1);

			_lstm_concat[idx].init(opts.characterSize * 2, -1);
		}

		_lstm_left.init(&model.lstm_left_param, opts.dropProb, true);
		_lstm_right.init(&model.lstm_right_param, opts.dropProb, false);


		for (int i = 0; i < _NonLinear_l.size(); i++) {
			_NonLinear_l[i].setParam(&model.NonLinear);
			_NonLinear_l[i].init(opts.characterSize , opts.dropProb);
			_NonLinear_r[i].setParam(&model.NonLinear);
			_NonLinear_r[i].init(opts.characterSize, opts.dropProb);

		}

		for (int i = 0; i < _character.size(); i++) {

			_character[i].setParam(&model.character);
			_character[i].init(opts.characterSize*2, opts.dropProb);
		}
		for (int i = 0; i < _neural_outputs.size(); i++) {
		
			_neural_outputs[i].setParam(&model.olayer_linear);
			_neural_outputs[i].init(opts.labelSize, -1);
		}


		//_neural_output.setParam(&model.olayer_linear);
		//_neural_output.init(opts.labelSize, -1);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Example exam, bool bTrain = false){
		_pcg->train = bTrain;
		// second step: build graph
		//forward 

		int char_num = exam.m_chars.size();
		if (char_num > max_sentence_length)
			char_num = max_sentence_clength;
		for (int i = 0; i < char_num; i++) {
			_char_inputs[i].forward(_pcg, exam.m_chars[i]);
			_char_inputs_r[i].forward(_pcg, exam.m_chars[i]);
			_char_concat[i].forward(_pcg, &_char_inputs[i], &_char_inputs_r[i]);

			_l_bichar_inputs[i].forward(_pcg, exam.m_bichars_l[i]);
			_l_bichar_inputs_r[i].forward(_pcg, exam.m_bichars_l[i]);
			_l_bichar_concat[i].forward(_pcg, &_l_bichar_inputs[i], &_l_bichar_inputs_r[i]);

			_l_concat_inputs[i].forward(_pcg, &_char_concat[i], &_l_bichar_concat[i]);

			_r_bichar_inputs[i].forward(_pcg, exam.m_bichars_r[i]);
			_r_bichar_inputs_r[i].forward(_pcg, exam.m_bichars_r[i]);
			_r_bichar_concat[i].forward(_pcg, &_r_bichar_inputs[i], &_r_bichar_inputs_r[i]);

			_r_concat_inputs[i].forward(_pcg, &_char_concat[i], &_r_bichar_concat[i]);
		}
	

		for (int i = 0; i < char_num; i++) {
			_NonLinear_l[i].forward(_pcg, &_l_concat_inputs[i]);
			_NonLinear_r[i].forward(_pcg, &_r_concat_inputs[i]);
		}


		_lstm_left.forward(_pcg, getPNodes(_NonLinear_l, char_num));
		_lstm_right.forward(_pcg, getPNodes(_NonLinear_r, char_num));
		for (int i = 0; i < char_num; i++) {
			_lstm_concat[i].forward(_pcg, &_lstm_left._hiddens[i], &_lstm_right._hiddens[i]);
		}

		for (int i = 0; i < char_num; i++) {
			_character[i].forward(_pcg, &_lstm_concat[i]);
		}

		
		for (int i = 0; i < char_num; i++) {
			_neural_outputs[i].forward(_pcg, &_character[i]);
		}

	}
		//_avg_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		//_max_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		//_min_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		//_concat.forward(_pcg, &_avg_pooling, &_max_pooling, &_min_pooling);
		
};

#endif /* SRC_ComputionGraph_H_ */