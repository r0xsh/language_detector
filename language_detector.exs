defmodule LanguageDetector do
  @moduledoc """
  An optimized statistical language detector that uses letter frequencies.
  Includes streaming for large files and profile persistence.
  """

  @profile_dir "profiles"
  # 1MB chunks
  @chunk_size 1024 * 1024
  # Number of top words to track
  @top_words_count 100
  # Number of top n-grams to track
  @top_ngrams_count 150

  # Weights for different features
  @weights %{
    # Common words
    words: 0.3,
    # Single letters
    letters: 0.1,
    # 2-letter sequences
    bigrams: 0.2,
    # 3-letter sequences
    trigrams: 0.3
  }

  defmodule Profile do
    @moduledoc "Structure for language profiles"
    defstruct letter_frequencies: %{},
              common_words: %{},
              bigrams: %{},
              trigrams: %{}
  end

  @doc """
  Train the detector using text files from the data directory.
  Processes files in chunks to handle large datasets efficiently.
  """
  def train_all(data_dir \\ "data") do
    File.mkdir_p!(@profile_dir)

    data_dir
    |> File.ls!()
    |> Stream.filter(&String.ends_with?(&1, ".txt"))
    |> Task.async_stream(&process_language_file(&1, data_dir),
      timeout: :infinity,
      ordered: false
    )
    |> Enum.reduce(%{}, fn {:ok, {lang, profile}}, acc ->
      Map.put(acc, lang, profile)
    end)
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
    text = Enum.join(chunks) |> String.downcase()

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

  @doc """
  Process a single language file to extract all features.
  """
  def process_language_file(file, data_dir) do
    language = Path.rootname(file)
    file_path = Path.join(data_dir, file)

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
profiles = LanguageDetector.train_all()

# Later, load saved profiles
loaded_profiles = LanguageDetector.load_profiles()

# a list of sentences in french, spanish, english and basque
texts = [
  "Bonjour tout le monde !",
  "Hola mundo !",
  "Hello world !",
  "Konnichiwa sekai !",
  "Hola mundo!",
  "Konnichiwa sekai!",
  "Je suis à la maison",
  "Il y a du monde dehors",
  "This is an example sentence.",
  "This is another example sentence.",
  "This is yet another example sentence.",
  "Lo egin nahi dut.",
  "Zure zentzugabekeria ez da nire errua.",
  "Lagundu al dizut?",
  "Bost urte ditut.",
  "Dušana dut izena."
]

Enum.each(texts, fn text ->
  LanguageDetector.guess(text, loaded_profiles)
end)
