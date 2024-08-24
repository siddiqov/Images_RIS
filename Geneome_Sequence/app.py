from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Union

def dna_k_mers_generation(dna_sequence: str, k_mers_length: int) -> List[str]:
    """Generate k-mers from the DNA sequence.
    
    Args:
        dna_sequence (str): The original DNA sequence.
        k_mers_length (int): The length of each k-mer to generate.
    
    Returns:
        List[str]: A list of generated k-mers.
    """
    k_mers: List[str] = []  # List to store k-mers
    # Loop through the DNA sequence to create k-mers
    for i in range(len(dna_sequence) - k_mers_length + 1):
        # Extract the substring of length k_mers_length
        k_mer: str = dna_sequence[i:i + k_mers_length]
        k_mers.append(k_mer)
        print(f"k-mer {i+1}: {k_mer}")  # Show each k-mer generated
    return k_mers

def dna_k_mers_concatenation(k_mers_list: List[str]) -> List[str]:
    """Concatenate k-mers into a single string.
    
    Args:
        k_mers_list (List[str]): The list of k-mers generated.
    
    Returns:
        List[str]: A list containing a single concatenated string of k-mers.
    """
    concatenated_k_mers: str = ' '.join(k_mers_list)  # String of concatenated k-mers
    print(f"\nConcatenated k-mers string: {concatenated_k_mers}")
    return [concatenated_k_mers]

def dna_k_mers_vectorization(k_mers_concatenated_list: List[str]) -> List[List[int]]:
    """Vectorize the concatenated k-mers using a bag-of-words model.
    
    Args:
        k_mers_concatenated_list (List[str]): A list containing the concatenated k-mers string.
    
    Returns:
        List[List[int]]: A 2D list representing the vectorized form of k-mers.
    """
    vectorizer: CountVectorizer = CountVectorizer()  # Bag-of-words model
    k_mers_vector: List[List[int]] = vectorizer.fit_transform(k_mers_concatenated_list).toarray().tolist()
    print(f"\nVectorized k-mers (using CountVectorizer): {k_mers_vector}")
    return k_mers_vector

def dna_bag_of_words(k_mers_concatenated_list: List[str]) -> List[List[int]]:
    """Alternative vectorization using a manual bag-of-words approach.
    
    Args:
        k_mers_concatenated_list (List[str]): A list containing the concatenated k-mers string.
    
    Returns:
        List[List[int]]: A 2D list representing the manually vectorized form of k-mers.
    """
    k_mers: List[str] = k_mers_concatenated_list[0].split()  # Split the concatenated string into k-mers
    k_mers_count: Counter = Counter(k_mers)  # Count occurrences of each k-mer
    vector: List[int] = [k_mers_count[k_mer] for k_mer in sorted(k_mers_count)]
    print(f"\nVectorized k-mers (using manual Bag of Words): {vector}")
    return [vector]


def main():
    
    # DNA sequence to process
    dna_sequence: str = "AACTTCTCCAACGACATCATGCTACTGCAGGTCAGGCACACTCCTGCCACTCTTG"
    print("DNA sequence string:\n{}".format(dna_sequence))

    # 1. K-mers Generation
    k_mers_length: int = 3
    print(f"\nGenerating k-mers of length {k_mers_length}...\n")
    dna_k_mers_generate_list: List[str] = dna_k_mers_generation(dna_sequence, k_mers_length)
    print("\nAll k-mers generated:\n{}".format(dna_k_mers_generate_list))

    # 2. K-mers Concatenation
    print("\nConcatenating k-mers...\n")
    dna_k_mers_concatenate_list: List[str] = dna_k_mers_concatenation(dna_k_mers_generate_list)

    # 3. K-mers Vectorization using CountVectorizer
    print("\nVectorizing k-mers...\n")
    dna_k_mers_vectorizer_array: List[List[int]] = dna_k_mers_vectorization(dna_k_mers_concatenate_list)

    # 4. Alternative Vectorization using Bag of Words
    print("\nAlternative vectorization...\n")
    dna_k_mers_vectorizer_array_alt: List[List[int]] = dna_bag_of_words(dna_k_mers_concatenate_list)


if __name__ == '__main__':
    main()
