defmodule LanguageDetector do
  def train_all(data_dir \\ "data") do
    data_dir
    |> File.ls!()
    |> Enum.filter(&String.ends_with?(&1, ".txt"))
    |> Enum.map(fn file ->
      language = Path.rootname(file)
      profile = calculate_frequencies(File.read!(Path.join(data_dir, file)))
      {language, profile}
    end)
    |> Map.new()
  end

  def calculate_frequencies(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^a-z]/, "")
    |> String.graphemes()
    |> Enum.frequencies()
    |> calculate_percentages()
  end

  defp calculate_percentages(frequencies) do
    total = frequencies |> Map.values() |> Enum.sum()

    frequencies
    |> Enum.map(fn {letter, count} ->
      {letter, count / total * 100}
    end)
    |> Map.new()
  end

  def detect(text, profiles) do
    text_freq = calculate_frequencies(text)

    distances =
      profiles
      |> Enum.map(fn {language, profile} ->
        {language, calculate_distance(text_freq, profile)}
      end)
      |> Map.new()

    confidence_scores = calculate_confidence(distances)
    most_likely = Enum.min_by(distances, fn {_lang, dist} -> dist end) |> elem(0)

    {most_likely, confidence_scores}
  end

  defp calculate_distance(freq1, freq2) do
    all_letters = MapSet.union(MapSet.new(Map.keys(freq1)), MapSet.new(Map.keys(freq2)))

    all_letters
    |> Enum.map(fn letter ->
      f1 = Map.get(freq1, letter, 0)
      f2 = Map.get(freq2, letter, 0)
      :math.pow(f1 - f2, 2)
    end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp calculate_confidence(distances) do
    max_distance = distances |> Map.values() |> Enum.max()

    distances
    |> Enum.map(fn {lang, dist} ->
      {lang, 100 * (1 - dist / max_distance)}
    end)
    |> Map.new()
  end
end

# Example usage:
profiles = LanguageDetector.train_all()

{language, confidence} =
  LanguageDetector.detect("Atenastik ezkutuan irtetea adostu dugu.", profiles)

IO.puts("\nDetected language: #{language}")
IO.puts("\nConfidence scores:")

confidence
|> Enum.each(fn {lang, score} ->
  IO.puts("#{lang}: #{Float.round(score, 2)}%")
end)
