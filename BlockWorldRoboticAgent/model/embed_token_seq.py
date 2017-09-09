import tensorflow as tf
import nltk
import numpy


class EmbedTokenSeq:
    """ Embeds a sequence of token using recurrent neural network with LSTM units."""

    # Token representing no words or NULL
    null = "$NULL$"

    # Token representing unknown words
    unk = "$UNK$"

    def __init__(self, output_size, ignore_case=True, num_steps=82, create_copy=None, scope_name="RNN"):
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.num_steps = num_steps
        self.lstm_size = 200
        self.output_size = output_size
        self.ignore_case = ignore_case

        # Word embedding matrix
        # Read the embedding matrix from file
        if create_copy is not None:
            # Read vocabulary, input_size and embedding matrix of parent rnn
            self.vocabulary = create_copy.vocabulary
            self.input_size = create_copy.input_size
            self.embedding = create_copy.embedding
            self.vocabulary_size = create_copy.vocabulary_size
        else:
            # (vocabulary, word_dim, word_embedding_matrix) = EmbedTokenSeq.read_word_embedding_matrix_from_file()
            word_dim = 150
            vocabulary = [EmbedTokenSeq.null, EmbedTokenSeq.unk]
            word_embedding_matrix = [[0] * word_dim] + [[0] * word_dim]
            (vocabulary, word_dim, word_embedding_matrix) = \
                EmbedTokenSeq.add_non_word2vec_words(vocabulary, word_dim, word_embedding_matrix)
            self.input_size = word_dim
            self.vocabulary = {}
            i = 0
            for token in vocabulary:
                self.vocabulary[token] = i
                i += 1

            self.vocabulary_size = len(vocabulary)
            self.embedding = tf.Variable(tf.convert_to_tensor(word_embedding_matrix, name="embeddingMatrix"))

        # Indices of words
        self.input_data = tf.placeholder(tf.int32, [None, self.num_steps])

        # Zero masking the irrelevant data
        self.mask = tf.placeholder(tf.float32, [None, self.num_steps])

        # Convert word indices into embeddings
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.output_size, forget_bias=1.0)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])

        self._initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self._initial_state
        with tf.variable_scope(scope_name):
            for time_step in range(self.num_steps):
                if time_step > 0 or create_copy is not None:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                zero_mask = self.mask[:, time_step]
                zero_mask = tf.reshape(zero_mask, [self.batch_size, 1])
                masked_output = tf.mul(cell_output, zero_mask)
                outputs.append(masked_output)

        # need to think of some other way here since longer sentences get more contribution.
        # cannot do reduce_mean as it will always divide by num_steps irrespective of
        # sentence length
        temporal_sum = tf.reduce_sum(outputs, 0)
        num_tokens = tf.reduce_sum(self.mask, 1)
        num_tokens = tf.reshape(num_tokens, [self.batch_size, 1])
        self.output = tf.div(temporal_sum, num_tokens)

        # Compute the set of all RNN variables for use in Q-learning which
        # requires copying variables from one network to another.
        self.all_variables = [self.embedding]
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name):
            self.all_variables.append(var)

        # Create mask.
        self.mask_ls = []
        for i in range(0, self.num_steps + 1):
            maski = [[1.0] * i + [0.0] * (self.num_steps - i)]
            self.mask_ls.append(maski)

        print "Created Token Seq Embedder"

    def get_input(self):
        return self.input_data

    def get_output(self):
        return self.output

    def get_max_time_step(self):
        return self.num_steps

    def get_zero_mask(self):
        return self.mask

    def get_batch_size(self):
        return self.batch_size

    def convert_text_to_indices(self, text):

        # Tokenize the text
        token_seq = nltk.word_tokenize(text)

        # Convert token sequence into indices
        indices = self.convert_token_seq_to_indices(token_seq)

        return indices

    def convert_token_seq_to_indices(self, token_seq):
        """ Converts token sequence to individual word indices
         and padds with NULL token (0 index) till max step
        TODO ask whether to ignore case """

        indices = []

        for token in token_seq:
            if self.ignore_case:
                ltoken = token.lower()
            else:
                ltoken = token
            if ltoken in self.vocabulary:
                indices.append(self.vocabulary[ltoken])
            else:
                indices.append(self.vocabulary[self.unk])

        return indices

    def pad_indices(self, indices):
        """ Pad the indices with NULL token till max step. Throw error
         if the indices length violets the max step already. """

        indices = indices + [self.vocabulary[self.null]] * (self.num_steps - len(indices))
        return indices

    def get_word_indices_and_mask(self, text):
        """ For each instruction, it tokenizes it and returns indices of every token in it.
        It addition, it pads the indices and also returns the mask indicating how many tokens exist. """

        indices = self.convert_text_to_indices(text)
        num_tk = len(indices)
        indices = self.pad_indices(indices)

        return [indices], self.mask_ls[num_tk]

    def pad_and_return_mask(self, indices):
        """ Given indices, pads the indices and also returns the mask indicating how many tokens exist. """
        num_tk = len(indices)
        indices = self.pad_indices(indices)

        return [indices], self.mask_ls[num_tk]

    @staticmethod
    def read_word_embedding_matrix_from_file():
        """ Reads word2vec word embedding matrix from a file.
        TODO move the file name above """

        lines = open("../word_embeddings/GoogleNews_word2vec.txt").readlines()
        word_embeddings = []
        vocabulary = []

        word_dim = None

        for line in lines:
            splits = line.split(":")
            word = splits[0]
            vector = [float(w) for w in splits[1].split(",")]

            if word_dim is None:
                word_dim = len(vector)
            elif not (word_dim == len(vector)):
                raise AssertionError("Word Dimensions inconsistent: " + str(word_dim) + " vs " + str(len(vector)))

            word_embeddings.append(vector)
            vocabulary.append(word)

        # Add null vector for padding and an unknown token
        # The 0 index is reserved for the zero vector padding representing NULL
        # The 1 index is reserved for unseen words
        # Embedding of unknown and null word is the zero vector.
        # Currently unknown words are not encountered during training hence this seems natural
        # The null word embedding does not affect result due to zero masking.
        vocabulary = [EmbedTokenSeq.null] + [EmbedTokenSeq.unk] + vocabulary
        word_embeddings = [[0] * word_dim] + [[0] * word_dim] + word_embeddings

        return vocabulary, word_dim, word_embeddings

    @staticmethod
    def normal_word_vector(word_dim):
        """ Returns a vector initialized by values from Standard normal distribution
        """

        vector = []
        val = numpy.random.normal(0.0, 1.0, word_dim)
        for i in range(0, word_dim):
            vector.append(val[i])
        return vector

    @staticmethod
    def add_non_word2vec_words(vocabulary, word_dim, word_embeddings):
        """ Not all words from training data are present in the word2vec.
        TODO move file name above
        """

        tokens = open("../BlockWorldSimulator/Assets/vocab_both").readlines()
        print "Read " + str(len(tokens)) + " many tokens from ../BlockWorldSimulator/Assets/ "
        size = len(vocabulary)

        for token in tokens:
            token = token.rstrip()
            if not (token in vocabulary):
                # create a vector of size word_dim initialize from normal distribution
                word_vector = EmbedTokenSeq.normal_word_vector(word_dim)

                # add the token and its word vector
                vocabulary.append(token)
                word_embeddings.append(word_vector)

        print "Vocabulary expanded from size " + str(size) + " to " + str(len(vocabulary))

        return vocabulary, word_dim, word_embeddings

    def copy_variables_to(self, other):

        if len(self.all_variables) != len(other.all_variables):
            raise AssertionError("Number of variables in the two embed_token_seq are different.")

        ops = []
        for (var_self, var_other) in zip(self.all_variables, other.all_variables):
            op = var_other.assign(var_self)
            ops.append(op)

        return ops
