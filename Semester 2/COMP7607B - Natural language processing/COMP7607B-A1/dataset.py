from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Word2VecConfig:
    window_size: int = 5
    min_freq: int = 5
    max_vocab_size: int = 10000


class Vocabulary:
    def __init__(self, config: Word2VecConfig):
        self.config = config
        self.word_counts = Counter[str]()
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def build_from_file(self, file_path: Path) -> None:
        """
        Builds vocabulary from text file.

        This method reads the text file line by line, splits each line into words,
        and counts the frequency of each word. It then creates word-to-index and
        index-to-word mappings from the counted words.

        Args:
            file_path: Path to the text file.

        Returns:
            None
        """
        print("Building vocabulary...")
        self._count_words(file_path)
        self._create_mappings()
        print(f"Vocabulary size: {len(self.word2idx)}")

    def _count_words(self, file_path: Path) -> None:
        """
        Counts word frequencies in the text file.

        This method reads the text file line by line and splits each line into words.
        It then updates the word counts dictionary with the words and their counts.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Split the line into words and update the word counts
                words = line.strip().lower().split()
                self.word_counts.update(words)

    def _create_mappings(self) -> None:
        """
        Creates word-to-index and index-to-word mappings from the counted words.

        The mappings are created by selecting the most common words that
        have a count greater than or equal to the minimum frequency
        specified in the configuration.
        """
        # Write Your Code Here
        most_common_words = [
            word for word, count in self.word_counts.most_common(self.config.max_vocab_size)
            if count >= self.config.min_freq
        ]
        self.word2idx = {word: idx for idx, word in enumerate(most_common_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}


class TrainingDataCreator:
    def __init__(self, config: Word2VecConfig, vocabulary: Vocabulary):
        self.config = config
        self.vocabulary = vocabulary

    def create_from_file(self, file_path: Path) -> List[Tuple[int, int]]:
        """
        Creates training data from the given text file.

        This method reads the text file line by line and processes each line
        to generate training examples. The training examples are stored in a
        list and returned at the end.

        Args:
            file_path: The path to the text file containing the corpus.

        Returns:
            A list of tuples, where each tuple contains the index of a target word
            and the index of a context word.
        """
        print("Creating training data...")
        training_data: List[Tuple[int, int]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Process the line and add the resulting training examples to the list
                training_data.extend(self._process_line(line))

        print(f"Number of training examples: {len(training_data)}")
        return training_data

    def _process_line(self, line: str) -> List[Tuple[int, int]]:
        """
        Processes a single line of text and returns training examples.

        Args:
            line: A line of text to process.

        Returns:
            A list of tuples, where each tuple contains the index of a target word
            and the index of a context word.
        """
        # Split the line into words and filter out words that are not in the vocabulary
        words = [
            word
            for word in line.strip().lower().split()
            if word in self.vocabulary.word2idx
        ]
        # Initialize an empty list to store the training examples
        examples: List[Tuple[int, int]] = []

        # Iterate over the words in the line
        for i, word in enumerate(words):
            # Get the index of the target word
            target_idx = self.vocabulary.word2idx[word]
            # Get the indices of the context words for the current target word
            context_indices = self._get_context_indices(words, i)
            # Add the training examples to the list
            examples.extend((target_idx, ctx_idx) for ctx_idx in context_indices)

        # Return the list of training examples
        return examples

    def _get_context_indices(self, words: List[str], target_pos: int) -> List[int]:
        """
        Gets indices of context words for a target position.

        For a given target word at position `target_pos`, this method returns a list
        of indices of context words. The context words are the words within the
        specified window size around the target word. The method ignores the target
        word itself and any words that are not in the vocabulary.

        Args:
            words: List of words in the corpus.
            target_pos: Position of the target word in the corpus.

        Returns:
            A list of indices of context words.
        """
        # Write Your Code Here
        left = max(0, target_pos - self.config.window_size)
        right = min(len(words), target_pos + self.config.window_size + 1)
        return [
            self.vocabulary.word2idx[words[i]]
            for i in range(left, right) if i != target_pos
        ]

class Word2VecDataset(Dataset):
    def __init__(self, text_file: str, config: Word2VecConfig = Word2VecConfig(), device: torch.device = torch.device("cpu")):
        file_path = Path(text_file)
        self.vocabulary = Vocabulary(config)
        self.vocabulary.build_from_file(file_path)

        creator = TrainingDataCreator(config, self.vocabulary)
        data_list = creator.create_from_file(file_path)
        self.data = torch.tensor(data_list, dtype=torch.long).to(device)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target, context = self.data[idx]
        return target, context
    
    def __len__(self) -> int:
        return len(self.data)


def get_loader(
    text_file: str,
    batch_size: int,
    window_size: int = 5,
    min_freq: int = 5,
    max_vocab_size: int = 10000,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 2,
) -> DataLoader:
    config = Word2VecConfig(window_size, min_freq, max_vocab_size)
    dataset = Word2VecDataset(text_file, config, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader
