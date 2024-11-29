Mix.install([{:unidecode, "~> 1.0"}])

defmodule LanguageDetector do
  @moduledoc """
  An optimized statistical language detector that uses letter frequencies.
  Includes streaming for large files and profile persistence.
  """

  @profile_dir "profiles"
  # 2MB chunks
  @chunk_size 1024 * 1024 * 2
  # Number of top words to track
  @top_words_count 300
  # Number of top n-grams to track
  @top_ngrams_count 400

  # Weights for different features
  @weights %{
    # Common words
    words: 0.35,
    # Single letters
    letters: 0.15,
    # 2-letter sequences
    bigrams: 0.25,
    # 3-letter sequences
    trigrams: 0.25
  }

  defmodule Profile do
    @moduledoc "Structure for language profiles"
    defstruct letter_frequencies: %{},
              common_words: %{},
              bigrams: %{},
              trigrams: %{}
  end

  def train_all(data_dir \\ "data") do
    File.mkdir_p!(@profile_dir)

    if is_binary(data_dir) do
      # Get all .txt files from the directory
      data_dir
      |> File.ls!()
      |> Stream.filter(&String.ends_with?(&1, ".txt"))
      |> Task.async_stream(
        fn file ->
          language = Path.rootname(file)
          file_path = Path.join(data_dir, file)
          process_training_file(language, file_path)
        end,
        timeout: :infinity,
        ordered: false
      )
      |> Enum.reduce(%{}, fn {:ok, {lang, profile}}, acc ->
        Map.put(acc, lang, profile)
      end)
    end
  end

  def guess(text, loaded_profiles) do
    {language, confidence} = LanguageDetector.detect(text, loaded_profiles)

    IO.puts("------------------------------------")
    IO.puts("\nText: #{text}")
    IO.puts("\nDetected language: #{language}")
    IO.puts("\nConfidence scores:")

    confidence
    |> Enum.sort_by(fn {_, score} -> score end, :desc)
    |> Enum.each(fn {lang, score} ->
      IO.puts("#{lang}: #{Float.round(score, 2)}%")
    end)
  end

  @doc """
  Generate n-grams from text.
  """
  def generate_ngrams(text, n) do
    text
    |> String.downcase()
    |> String.replace(~r/[^a-z]/, "")
    |> String.graphemes()
    |> Stream.chunk_every(n, 1, :discard)
    |> Stream.map(&Enum.join/1)
    |> Enum.frequencies()
  end

  @doc """
  Process a chunk of text and count frequencies of all features.
  """
  def process_chunk(chunks) when is_list(chunks) do
    text = Enum.join(chunks) |> String.downcase() |> Unidecode.decode()

    # Get letter frequencies
    letter_freqs =
      text
      |> String.replace(~r/[^a-z]/, "")
      |> String.graphemes()
      |> Enum.frequencies()

    # Get word frequencies
    word_freqs =
      text
      |> String.split(~r/[^a-z0-9]+/, trim: true)
      |> Enum.filter(&(String.length(&1) >= 4))
      |> Enum.frequencies()

    # Get n-gram frequencies
    bigram_freqs = generate_ngrams(text, 2)
    trigram_freqs = generate_ngrams(text, 3)

    {letter_freqs, word_freqs, bigram_freqs, trigram_freqs}
  end

  @doc """
  Merge frequency maps from different chunks.
  """
  def merge_frequencies({l1, w1, b1, t1}, {l2, w2, b2, t2}) do
    {
      merge_maps(l1, l2),
      merge_maps(w1, w2),
      merge_maps(b1, b2),
      merge_maps(t1, t2)
    }
  end

  defp merge_maps(map1, map2) do
    Map.merge(map1, map2, fn _k, v1, v2 -> v1 + v2 end)
  end

  defp process_training_file(language, file_path) do
    {letter_freqs, word_freqs, bigram_freqs, trigram_freqs} =
      file_path
      |> File.stream!([], @chunk_size)
      |> Stream.chunk_every(10)
      |> Stream.map(&process_chunk/1)
      |> Enum.reduce({%{}, %{}, %{}, %{}}, &merge_frequencies/2)

    profile = %Profile{
      letter_frequencies: calculate_percentages(letter_freqs),
      common_words: get_top_items(word_freqs, @top_words_count),
      bigrams: get_top_items(bigram_freqs, @top_ngrams_count),
      trigrams: get_top_items(trigram_freqs, @top_ngrams_count)
    }

    save_profile(language, profile)
    {language, profile}
  end

  defp calculate_percentages(frequencies) do
    total = frequencies |> Map.values() |> Enum.sum()

    case total do
      0 ->
        %{}

      _ ->
        frequencies
        |> Stream.map(fn {key, count} ->
          {key, count / total * 100}
        end)
        |> Map.new()
    end
  end

  defp get_top_items(freqs, count) do
    total = freqs |> Map.values() |> Enum.sum()

    case total do
      0 ->
        %{}

      _ ->
        freqs
        |> Enum.sort_by(fn {_, count} -> count end, :desc)
        |> Enum.take(count)
        |> Enum.map(fn {item, count} ->
          {item, count / total * 100}
        end)
        |> Map.new()
    end
  end

  @doc """
  Calculate cosine similarity between two frequency distributions.
  """
  def calculate_similarity(freq1, freq2) do
    # Calculate dot product
    dot_product =
      freq1
      |> Map.keys()
      |> Stream.concat(Map.keys(freq2))
      |> Enum.uniq()
      |> Enum.reduce(0, fn key, acc ->
        v1 = Map.get(freq1, key, 0)
        v2 = Map.get(freq2, key, 0)
        acc + v1 * v2
      end)

    # Calculate magnitudes
    magnitude1 =
      freq1
      |> Map.values()
      |> Enum.reduce(0, fn v, acc -> acc + v * v end)
      |> :math.sqrt()

    magnitude2 =
      freq2
      |> Map.values()
      |> Enum.reduce(0, fn v, acc -> acc + v * v end)
      |> :math.sqrt()

    # Return cosine similarity
    case {magnitude1, magnitude2} do
      {0.0, _} -> 0.0
      {_, 0.0} -> 0.0
      {m1, m2} -> dot_product / (m1 * m2)
    end
  end

  @doc """
  Detect the language of the given text using all features.
  """
  def detect(text, profiles) do
    # Process input text
    {letter_freqs, word_freqs, bigram_freqs, trigram_freqs} = process_text(text)

    input_features = %{
      letters: calculate_percentages(letter_freqs),
      words: calculate_percentages(word_freqs),
      bigrams: calculate_percentages(bigram_freqs),
      trigrams: calculate_percentages(trigram_freqs)
    }

    # Calculate combined scores
    scores =
      profiles
      |> Stream.map(fn {language, profile} ->
        feature_scores = %{
          letters: calculate_similarity(input_features.letters, profile.letter_frequencies),
          words: calculate_similarity(input_features.words, profile.common_words),
          bigrams: calculate_similarity(input_features.bigrams, profile.bigrams),
          trigrams: calculate_similarity(input_features.trigrams, profile.trigrams)
        }

        # Combine scores with weights
        weighted_score =
          @weights
          |> Enum.map(fn {feature, weight} ->
            Map.get(feature_scores, feature, 0) * weight
          end)
          |> Enum.sum()

        {language, weighted_score}
      end)
      |> Map.new()

    # Calculate confidence scores
    confidence_scores = calculate_confidence(scores)
    most_likely = Enum.max_by(scores, fn {_lang, score} -> score end) |> elem(0)

    {most_likely, confidence_scores}
  end

  defp process_text(text) when byte_size(text) > @chunk_size do
    text
    |> Stream.unfold(&split_chunk/1)
    |> Stream.map(&process_chunk([&1]))
    |> Enum.reduce({%{}, %{}, %{}, %{}}, &merge_frequencies/2)
  end

  defp process_text(text), do: process_chunk([text])

  defp split_chunk(<<chunk::binary-size(@chunk_size), rest::binary>>), do: {chunk, rest}
  defp split_chunk(<<>>), do: nil
  defp split_chunk(remainder), do: {remainder, <<>>}

  defp calculate_confidence(scores) do
    max_score = scores |> Map.values() |> Enum.max()
    min_score = scores |> Map.values() |> Enum.min()
    range = max_score - min_score

    scores
    |> Stream.map(fn {lang, score} ->
      normalized = if range == 0, do: 100, else: (score - min_score) / range * 100
      {lang, normalized}
    end)
    |> Map.new()
  end

  # Profile storage functions remain the same
  def save_profile(language, profile) do
    profile_path = Path.join(@profile_dir, "#{language}.profile")
    binary = :erlang.term_to_binary(profile)
    File.write!(profile_path, binary)
  end

  def load_profiles do
    @profile_dir
    |> File.ls!()
    |> Stream.filter(&String.ends_with?(&1, ".profile"))
    |> Task.async_stream(&load_single_profile/1)
    |> Enum.reduce(%{}, fn {:ok, {lang, profile}}, acc ->
      Map.put(acc, lang, profile)
    end)
  end

  def load_single_profile(file) do
    language = Path.rootname(file)
    profile_path = Path.join(@profile_dir, file)

    profile =
      profile_path
      |> File.read!()
      |> :erlang.binary_to_term()

    {language, profile}
  end
end

# Train and save profiles
_profiles = LanguageDetector.train_all("data")

# Later, load saved profiles
loaded_profiles = LanguageDetector.load_profiles()

texts = [
  # English - Mix of common structures, idioms, and varying complexity
  "The weather is lovely today.",
  "I can't believe how fast time flies!",
  "Would you mind helping me with this?",
  "She's been working here for five years.",
  "The quick brown fox jumps over the lazy dog.",
  "Could you please pass me the salt?",
  "That movie was absolutely fantastic!",
  "I'll think about it and get back to you.",
  "We should probably get going soon.",
  "Have you ever been to Paris in springtime?",

  # French - Including contractions, accents, and common expressions
  "Je ne sais pas quoi faire aujourd'hui.",
  "Pourriez-vous m'indiquer le chemin ?",
  "C'est vraiment une belle journée !",
  "J'aimerais un café, s'il vous plaît.",
  "Nous sommes allés au cinéma hier soir.",
  "Il fait beau temps ce matin.",
  "Je dois partir travailler maintenant.",
  "Enchantée de faire votre connaissance.",
  "Comment allez-vous aujourd'hui ?",
  "Je voudrais réserver une table pour deux.",

  # Spanish - Including subjunctive, pronouns, and typical expressions
  "¿Qué tal has estado últimamente?",
  "Me gustaría un café con leche, por favor.",
  "No sé si podré ir a la fiesta mañana.",
  "¡Qué bonito día hace hoy!",
  "¿Has visto mis llaves por alguna parte?",
  "Necesito que me ayudes con esto.",
  "¿Podrías hablar más despacio?",
  "Vamos a la playa este fin de semana.",
  "¡Cuánto tiempo sin verte!",
  "Me encantaría conocer tu ciudad.",

  # Basque - Including specific grammatical structures and vocabulary
  "Gaur goizean mendira joan naiz.",
  "Euskara ikasten ari naiz.",
  "Etxera noa orain.",
  "Bihar goizean elkar ikusiko dugu.",
  "Zer moduz zaude?",
  "Kafea nahi duzu?",
  "Ez dakit zer egin.",
  "Nire izena Mikel da.",
  "Atzo zinera joan ginen.",
  "Euskal Herrian bizi naiz."
]

Enum.each(texts, fn text ->
  LanguageDetector.guess(text, loaded_profiles)
end)
