# Statistical Language Detector

Very naive and simple statistical language detector in Elixir using [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance) and letter frequencies.

## TODO

- Try another approach using k-nearest neighbors algorithm
- Compile language profiles into `[language].profile` files

## Limitations
- Basic statistical approach using only letter frequencies
- No support for special characters or n-grams
- Memory intensive for very large datasets despite chunking

## Usage

```bash
elixir language_detector.exs
```

## Sample Data
You can get sample data from: https://tatoeba.org
