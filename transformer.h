//~/pulp_box/forejune_co_cuda/transformer6/transformer.h
#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include "util.h"
using namespace std;
class Tokenizer {
private:
    unordered_map<string, int> words;
    unordered_map<int, string> wordIndex;
    int nTokens;        // number of words
    // special tokens
    const string PAD = "<PAD>";
    const string UNK = "<UNK>";
    const string BOS = "<BOS>";
    const string EOS = "<EOS>";
    const string SEP = "<SEP>";
    void addWord(const string &w) {
        if (words.find(w) == words.end()) {
            int id = nTokens;
            words[w] = id;
            wordIndex[id] = w;
            nTokens++;
        }
    }
    vector<string> split(const string &str) {
        vector<string> tokens;
        stringstream ss(str);
        string word;
        while (ss >> word) {  tokens.push_back(word);  }
        return tokens;
    }
public:
    Tokenizer();
    Tokenizer(ifstream &fs);
    unordered_map<string, int>  getTokensWords();
    vector<int> tokenize(const string& str);
    string detokenize(const vector<int>& tokenIDs);
    int get_nTokens() const;
    unordered_map<int, string> get_wordIndex();
};
class Embedding {
private:
    int nTokens;        // number of words
    int dim;            // embedding dimension
    matrixd em_matrix;  // embedding matrix;
public:
    Embedding(int sequence_length, int embeddingDimension);
    // create embedding matrix with tokenIDs
    matrixd embed(const vector<int>& tokenIDs);
    void backProp(const vector<int>& tokens, const matrixd& dX, double eta);
    int get_embedDim() const;
    int get_nTokens() const;
};
// Positional Encoding
class PositionalEncoding {
private:
    int maxTokens; // maximum sequence length
    int dModel;    // embedding dimension
    matrixd PE;    // positional_encodings matrix;
public:
    PositionalEncoding(int max_seq_length, int embedding_dimension);
    void extendPE(int new_maxTokens);
    matrixd addPE(const matrixd& embeddings);
    matrixd getPE();
};
// Add & Norm
struct AddNormGrad {
    matrixd dL_dA;  // residual path
    matrixd dL_dB;  // sublayer path
};
class AddnNorm {
private:
    vector<double> gamma;   // size dModel
    vector<double> beta;    // size dModel
    matrixd Zhat;           // normalized values (nt x d)
    vector<double> mean;    // per row (nt)
    vector<double> var;     // per row (nt)
    double eps;             // small value
public:
   AddnNorm(int dModel);
   // forward computation
   matrixd  forward (const matrixd &A, const matrixd &B);
   // Assume forward already filled: Zhat, mean, var
   AddNormGrad backProp(const matrixd &X, const matrixd &dL_dY, double eta);
   // Optional setters (to plug in forward results)
   void setCache(const matrixd& zhat, const vector<double>& m, const vector<double>& v);
    // Accessors (optional)
    const vector<double>& getGamma();
    const vector<double>& getBeta();  
};
// Feed-forward Network
class FeedForward {
private:
    matrixd W1, W2;
    int dModel, d_ff;
    // cache
    matrixd F1;   // after activation function ReLU
    matrixd Z;    // before ReLU (needed for derivative)
public:
    // constructor
    FeedForward(int dModel, int d_ff_);
    // Forward (for cache)
    matrixd FFoutput(const matrixd &X);
    // Backprop
    matrixd backProp(const matrixd &X, const matrixd &dL_dY,  double eta);
};
// ----------- MultiHeadAttention -----------
struct MHA_Gradient {
    matrixd dL_dQ;
    matrixd dL_dK;
    matrixd dL_dV;
};
class MultiHeadAttention {
private:
    int dModel, nHeads, d_k;
    vector<matrixd> WQ, WK, WV; // per head
    matrixd WO;
    // cache
    vector<matrixd> Qs, Ks, Vs, Ps, As;
public:
    MultiHeadAttention(int dModel, int nHeads);
    void clearCache();
    matrixd computeAttention(const matrixd &X_Q, const matrixd &X_K, const matrixd &X_V, bool mask);
    MHA_Gradient backProp( const matrixd &X_Q, const matrixd &X_K, const matrixd &X_V,
            const matrixd &dL_dH, double eta, bool mask);
};
// ----------------- OutputLayer ---------------
class OutputLayer {
private:
    matrixd W;   // dModel x vocab_size
    int dModel, vocab_size;
public:
    OutputLayer(int dModel, int vocab_size);
    matrixd forward(const matrixd &X);
    matrixd backward(const matrixd &X, const matrixd &P, const vector<int> &targets, double eta);
};
// ---------- EncoderLayer --------------------
class EncoderLayer {
private:
    MultiHeadAttention mha;
    FeedForward ffn;
    AddnNorm norm1, norm2;
    matrixd X, H1, H2;
public:
      EncoderLayer();   //default constructor
      EncoderLayer(int d_model, int nHeads, int d_ff);
      matrixd forward(const matrixd& input);
      matrixd backProp(const matrixd& dL_dOut, double eta);
};
// ------------ Decoder Layer -------------------
struct DecoderGrad {
    matrixd dL_dX;  // gradient to decoder input
    matrixd dL_dE;  // gradient to encoder output
};
class DecoderLayer {
private:
    FeedForward ffn;
    AddnNorm norm1, norm2, norm3;
    matrixd X, H1, H2, H3;
    matrixd E;    // final output of Encoder
public:
    MultiHeadAttention self_mha;
    MultiHeadAttention cross_mha;
    DecoderLayer(int d_model, int nHeads, int d_ff);
    matrixd forward(const matrixd& input, const matrixd& encoder_output);
     DecoderGrad backProp(const matrixd& dL_dOut, double eta);
};
// -------------------  Transformer ---------------
class MiniTrans {
private:
    Embedding embed;
    PositionalEncoding pe;
    vector<EncoderLayer> eLayers; 
    vector<DecoderLayer> dLayers;
    OutputLayer output;
    int seq_len, vocab_size;
    int dModel;
    matrixd X_embed;   // cache
    matrixd X_pe;      // cache
    matrixd X_final;
    matrixd dec_out;    // cache decoder output
public:
    MiniTrans(int vocab_size, int seq_len, int dModel, int nHeads, int d_ff, int nLayers);
    matrixd forward(const vector<int> &src_tokens, const vector<int> &target_tokens);
    double trainStep(const vector<int> &src, const vector<int> &target_input, 
            const vector<int> &target_out, double eta);
    vector<int> generate(const vector<int> &src, int max_len, int start_token, int end_token);
 };
#endif
