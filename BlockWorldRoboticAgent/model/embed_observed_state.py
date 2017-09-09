import embed_token_seq
import image_preprocessing
import embed_image

class EmbedObservedState:
    """ This provides a general non-task specific embedding of an image, instruction and
        embedding of status flag and previous action. The individual embeddings are generated
        and concatenated to generate embeddings for the observed agent state. """

    def __init__(self):

        # Neural network for embedding text
        self.n_text = 250
        self.text_embedder = embed_token_seq.EmbedTokenSeq(self.n_text)
        text_embedding = self.text_embedder.get_output()
        ####################
        # Create bucket network
        self.buckets = [15, 30, 45]
        self.embed_token_seq_buckets = []
        for bucket in self.buckets:
            embed_token_seq_bucket = \
                embed_token_seq.EmbedTokenSeq(self.n_text, num_steps=bucket, create_copy=self.text_embedder)
            self.embed_token_seq_buckets.append(embed_token_seq_bucket)
        ####################

        # Image Preprocessing
        self.image_preprocessor = image_preprocessing.ImagePreprocessing()

        # Neural network for embedding image
        self.n_image = 200
        self.image_embedder = embed_image.EmbedImage(self.n_image, image_dim)
        image_embedding = self.image_embedder.get_output()

        # Network for embedding past action
        # 6 actions, one for no-action
        self.n_status_flag_dim = 18
        self.n_direction_dim = 24
        self.n_previous_action_embedding = self.n_status_flag_dim + self.n_direction_dim
        self.null_previous_action = (2, 5)
        self.previous_action_embedder = epa.EmbedPreviousAction(3, self.n_status_flag_dim, 6, self.n_direction_dim)
        previous_action_embedding = self.previous_action_embedder.get_output()

        # Neural network for mixing the embeddings of text
        # and image and generate probabilities over block-ids and direction
        if self.train_alg == TrainingAlgorithm.SUPERVISEDMLE \
                or self.train_alg == TrainingAlgorithm.REINFORCE \
                or self.train_alg == TrainingAlgorithm.MIXER:
            use_softmax = True
        else:
            use_softmax = False
        self.mix_and_gen_prob = mix_and_gen_prob.MixAndGenerateProbabilities(
            self.n_text, self.n_image, self.n_previous_action_embedding,
            text_embedding, image_embedding, previous_action_embedding, 5, use_softmax)
        ####################
        self.mix_and_gen_prob_buckets = []
        for i in range(0, len(self.buckets)):
            mix_and_gen_prob_bucket = mix_and_gen_prob.MixAndGenerateProbabilities(
                self.n_text, self.n_image, self.n_previous_action_embedding,
                self.embed_token_seq_buckets[i].get_output(), image_embedding,
                previous_action_embedding, 5, use_softmax, create_copy=self.mix_and_gen_prob)
            self.mix_and_gen_prob_buckets.append(mix_and_gen_prob_bucket)
            ####################