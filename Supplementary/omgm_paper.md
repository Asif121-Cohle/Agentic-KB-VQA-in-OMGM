# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval

## Abstract
Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems.

## 1 Introduction
Visual Question Answering (VQA) involves answering questions about a given query image by comprehensively understanding its semantic content, demanding proficiency in both visual and textual understanding. Large Language Models (LLMs) have demonstrated remarkable generalization and reasoning capabilities in text-based tasks. By integrating visual encoders with LLMs, Multimodal Large Language Models (MLLMs) have emerged as an effective approach for handling VQA tasks, as they can jointly model both image and text representations for enhanced comprehension and reasoning. Knowledge-Based Visual Question Answering (KB-VQA) extends this challenge by requiring the incorporation of external world knowledge that transcends the visible elements of the image. In KB-VQA, questions are designed to probe for information pertaining to the image’s subject matter, but necessitate insights not directly present within the image itself.

Retrieval-Augmented Generation (RAG) offers a cost-effective and efficient solution to the challenges of KB-VQA by retrieving query-relevant knowledge from the knowledge base and integrating it as contextual information for response generation. The effectiveness of this approach depends on efficient retrieval mechanisms capable of identifying the most relevant information from large-scale, heterogeneous knowledge bases. However, multimodal retrieval in KB-VQA introduces complexities beyond the standard text-based retrieval used in most RAG systems. These complexities stem from two key factors:
*   **Multiple Modalities.** Both queries and knowledge bases comprise multiple modalities, such as images and text, necessitating diverse relevance assessment strategies that may involve single-modal, cross-modal, or multi-modal approaches.
*   **Hybrid Granularities.** Queries and knowledge bases often involve information at different levels of granularities. For instance, a query might include a query image with coarse-grained knowledge identifying a subject, paired with a question seeking fine-grained details about the subject. Similarly, a knowledge base might consist of articles with coarse-grained overview reference images and titles as well as fine-grained detailed sections containing in-depth information.

Recent research efforts have explored both single-step and multi-step multimodal retrieval strategies to enhance retrieval effectiveness and ultimately improve answer generation. Single-step approaches retrieve passages directly using a multimodal query. While effective, these methods often require expensive task-specific pretraining and incur high computational costs during inference due to exhaustive full-range searches. Conversely, multi-step retrieval methods adopt hierarchical retrieval strategies that enhance searching efficiency by progressively narrowing the search space. Nevertheless, the potential of multi-step retrieval remains underexplored. Current approaches frequently overlook the intricate interplay between retrieval modalities, granularities, and the sequencing of retrieval steps, limiting their overall effectiveness and adaptability.

We propose a multimodal RAG system, OMGM, which employs a coarse-to-fine, multi-step retrieval strategy to effectively Orchestrate Multiple Granularities and Modalities across queries and knowledge bases, enhancing multimodal retrieval. 

![Figure 1: An illustration of our method. First, a coarse-grained cross-modal entity search is performed between entity summaries and the query image to retrieve the top-k entity candidates. Next, a hybrid-grained multimodal fusion reranker uses the multimodal query to retrieve image-section pairs, refining the selection of the most relevant entity. Finally, within knowledge associated with the top-1 entity, fine-grained textual filtering is applied to extract most relevant section, which is used to enhance generation in downstream generator.](figure1_placeholder)

As illustrated in Figure 1, our system operates in three stages: it begins with a coarse-grained cross-modal retrieval to identify an initial pool of entity article candidates, followed by a hybrid-grained multimodal reranker that leverages both coarse-grained and fine-grained knowledge to rerank candidates and select the most relevant entity. Finally, a fine-grained text reranker filters the selected entity’s sections to extract the most pertinent sections for augmented response generation. Throughout the process, query and candidates are aligned based on their granularities, and embedding models are carefully selected to ensure effective multimodal representation. More importantly, these retrieval steps interact sequentially, with similarity scores from earlier stages propagated forward and fused in subsequent steps, enabling a cohesive and context-aware retrieval process.

## 2 Related Work

### 2.1 KB-VQA
Traditional VQA tasks focus primarily on the visual content within images to answer the related textual question. However, KB-VQA expands this by incorporating external knowledge bases to address questions requiring information beyond the image. Datasets such as OK-VQA and A-OKVQA involve visual question answering that require outside knowledge, which consists of general commonsense that lacks distinctiveness. The emergence of the E-VQA and InfoSeek datasets presents a greater challenge for end-to-end LLMs/MLLMs by incorporating varying granularities of encyclopedic knowledge alongside extensive multimodal information. In response to these characteristics of KB-VQA task, our work adopts a multi-step, granularity-aligned retrieval framework, resulting in improved retrieval and VQA performance compared to the current multimodal RAG system.

### 2.2 Vision-language RAG
Retrieval-Augmented Generation (RAG) enhances the generation performance of LLMs by retrieving external documents relevant to the input query and using them as guiding context in prompts. In addition to text augmentation, recent work has targeted generation enhancement specifically for vision-language tasks, which are more closely aligned with complex real-world scenarios. Vision-language retrieval-augmented generation typically entails handling multimodal documents and queries, requiring the retrieval process across these modalities. Unlike previous methods, our system constructs a multi-step retrieval process that integrates granularity alignment with cross-step joint retrieval. At each step, we align queries with the corpus using techniques such as summary extraction, multimodal fusion, and question-based section denoising, then fuse results via similarity fusion to boost recall and enhance downstream KB-VQA performance of MLLMs/LLMs.

## 3 Methodology
To address the challenges in KB-VQA, we introduce a multimodal RAG system that features efficient coarse-to-fine, multi-step multimodal retrieval. This system is capable of extracting pertinent information from a vast multimodal knowledge base with millions of entries. The retrieved data is subsequently used to enhance the generator’s responses. An overview of the proposed framework is shown in Figure 2, which comprises three key components: coarse-grained cross-modal entity searching, hybrid-grained multimodal-fused reranking, and fine-grained section-augmented generation.

![Figure 2: An overview of our framework. Our framework first performs a coarse-grained, cross-modal entity search by using offline-generated knowledge summaries as indices and a query image as the query; it then conducts a hybrid-grained multimodal-fused reranking that aligns image and text details via feature fusion and integrates cross-step similarities. Finally, a fine-grained section-augmented generation selects the most relevant knowledge section through combined text and multimodal similarities to support answer generation.](figure2_placeholder)

### 3.1 Coarse-Grained Cross-Modal Entity Searching
To preliminarily filter out wiki entity information related to the query from a large corpus, we design a coarse-grained entity retrieval method. In this step, the query image indicating a subject serves as a coarse-grained query, while entity summaries act as coarse-grained candidates, ensuring appropriate information granularity alignment.

**Summary Generation.** Since the complete wiki entity articles are too redundant as entity information for effective retrieval indexing, we align them with the macro-level entity information of the query image. Specifically, we use article summaries as the retrieval index for wiki entities, these concise abstracts densely encapsulate key aspects of the entity’s information. To maintain real-time retrieval efficiency, we generate summaries for all entity articles in the knowledge base offline. Given an entity article $a_i$, we employ a pre-trained language model $M_s$ with an instruction prompt template $P$ to generate a summary $s_i$, ensuring that the summary is both informative and well-aligned with the retrieval task.
$$ s_i = M_s(P, a_i) \quad (1) $$

**Image-to-Summary Entity Searching.** After obtaining the wiki entity summaries, we use the query image from the VQA triplet, uniquely representing the target entity, as the query object for entity searching. Specifically, both the query image and candidate entity summaries are transformed into feature vectors, which are used for similarity-based matching for entity searching.
$$ Ent_k = F(E_v(I_q), E_t(S), k) \quad (2) $$
where $I_q$ denotes the query image, and $S = \{s_1, s_2, ..., s_n\}$ represents all entity summaries in the knowledge base. These are encoded through the visual encoder $E_v$ and textual encoder $E_t$ of CLIP, respectively. We use the Faiss library $F$ to index the feature vectors of the entity summaries and perform embedding matching based on the inner product. Finally, we retain the top-$k$ most relevant entity summaries along with their corresponding entity information $Ent_k$.

### 3.2 Hybrid-Grained Multimodal-Fused Reranking
After the initial search, we obtain the top-$k$ entity candidates most similar to the query subject. Within this candidate set, we extract hybrid-grained multimodal fusion feature matrices from the coarse-grained images and fine-grained texts provided by the query and candidate knowledge. Then, by employing a late-interaction mechanism, we obtain a fine-grained section similarity. By integrating the entity similarity from the previous step, we derive a coarse-grained reranking similarity that yields a more accurate ordering of relevant entities.

**Multimodal Fusion Feature Matching.** Specifically, we leverage the Q-Former architecture to extract multimodal fusion features from both query and candidates. The pipeline for fusion feature extraction is as follows:
$$ Q = E_m(q), \quad C_{sec_h^e} = E_m(d_{sec_h^e}) \quad (3) $$
As the input to the q-former multimodal encoder $E_m$, we use the query image $I_q$ and textual question $T_q$ on the query side, forming the input pair $q=(I_q, T_q)$. On the candidate side, we construct the input as $d_{sec_h^e}=(I_e, sec_h^e)$, where $I_e$ is the main image of entity and $sec_h^e$ is the $h$-th section of the entity article $a_e=\{sec_1^e, sec_2^e,..., sec_p^e\}$. The fusion feature matrix $Q$ and $C_{sec_h^e}$ are the output vectors corresponding to the query tokens of encoder $E_m$. We calculate the fine-grained multimodal similarity $sim_{sec_h^e}^m$ between the query $q$ and candidate $d_{sec_h^e}$ using the late-interaction, which fully leverages the correlations between each token vector in the feature matrix and integrates them into a single similarity score through a Max-Sum operation. $l_Q$ and $l_C$ denote the total number of tokens in $Q$ and $C_{sec_h^e}$, respectively.
$$ sim_{sec_h^e}^m = sim_m(q, d_{sec_h^e}) = \sum_{i=1}^{l_Q} \max_{j=1}^{l_C} Q_i (C_{sec_h^e}^j)^\top \quad (4) $$
Finally, we obtain the coarse-grained entity similarity by maximizing $sim_m$ of all the sections in the entity article $a_e$, with section size $l_{a_e}$. We then compute the final reranking similarity scores for the top-$k$ entities $Ent_k$ by performing a weighted summation with the initial entity similarity $sim_e^c$ from the previous cross-modal retrieval step. The most relevant entity $e_{top1}$ is selected based on the highest reranking score. The weighting factor $\alpha$ balances the contributions of the two similarity measures, optimizing reranking performance.
$$ e_{top1} = \arg\max_{e \in Ent_k} \left( \alpha \cdot sim_e^c + (1-\alpha) \cdot \max_{h=1}^{l_{a_e}} sim_{sec_h^e}^m \right) \quad (5) $$

**Multimodal Reranker Training.** To train the multimodal fusion encoder, we employ contrastive learning with the hard negative samples from previous retrieval step. Specifically, the coarse-grained retrieval in earlier step generates a top-$k$ candidate entity set for each training sample. Most of these entities share similar coarse-grained characteristics, but differ in fine-grained details. To construct training pairs, we randomly select negative pairs by pairing the main images of candidate entities with non-evidentiary sections from their articles. In contrast, positive pairs consist of the main image and the evidence section of the correct entity.
$$ L = -\log \frac{\exp(sim_m(q, d^+)/T)}{\sum_{j=1}^N \exp(sim_m(q, d_j)/T)} \quad (6) $$
By constructing each training sample containing $N$ contrastive pairs $d$, we train the reranker to match queries with the positive candidate pairs $d^+$ based on hybrid-grained multimodal information with the adaptive temperature $T$ for the smoothness of the softmax distribution. This multimodal-fused retrieval approach not only complements the coarse-grained entity retrieval in step one but also paves the way for subsequent fine-grained refinement.

### 3.3 Fine-Grained Section-Augmented Generation
After completing the first two steps of entity ranking, we identify $e_{top1}$ as the entity most relevant to the query. In this step, we perform fine-grained knowledge filtering on the entity information and use the filtered knowledge as auxiliary context to enhance the generation of the downstream generator. Specifically, we employ a pre-trained textual reranker $R_t$ to calculate the textual similarities $sim_{sec}^t = R_t(T_q, sec)$ between sections of the entity article and question. These textual similarities are then combined with the fine-grained multimodal-fused similarity $sim_{sec}^m$ obtained from the previous step through weighted summation, enabling us to extract the most relevant entity section $sec_{best}^{e_{top1}}$ for the query.
$$ sec_{best}^{e_{top1}} = \arg\max_{sec \in e_{top1}} \left( \beta \cdot sim_{sec}^m + (1-\beta) \cdot sim_{sec}^t \right) \quad (7) $$
Here, $\beta$ serves as a balancing factor to weigh the similarities derived from retrieval at different steps. Finally, the most relevant section is provided as input context along with the query to the generator for question-answering.

## 4 Experiments

### 4.1 Datasets
We utilize two challenging KB-VQA datasets, InfoSeek and E-VQA, for training and testing. To ensure fairness in both retrieval and VQA evaluations, we adopt the same setup used by many previous studies, with the specific configuration detailed in Table 1. For the images associated with the entity articles in the knowledge base, we crawled multiple images, including the main image, from each entity’s corresponding wikipedia page to constitute the knowledge base’s image set. Our experiments primarily focus on assessing both retrieval and question-answer performance: retrieval is evaluated using Recall@K, while VQA performance is measured using the official metrics for each dataset (e.g., BEM score for E-VQA and both VQA accuracy and relaxed accuracy for InfoSeek).

| Datasets | #Samples | #Entity Articles | Gen. Train | Ret. Train | Valid | Test | Train | Valid/Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Infoseek | 100K | -- | 71,335 | 100K | 100K | | | |
| E-VQA | 100K | 191k | 11,696 | 4,750 | 2M | 2M | | |

*Table 1: Statistics of Infoseek and E-VQA datasets used in our experiments. Gen. Train and Ret. Train represent the number of training samples for the generator and the multimodal fusion module in reranking, respectively.*

### 4.2 Implementation Details
**Initial Entity Searching.** To produce high-quality summaries of entity articles, we leveraged LangGPT to develop the "Wiki Summary Generator Assistant" prompt. For efficient entity search using the FAISS library, we employed the pooled embeddings from the last layer of encoder for calculating image-text similarity. We set $k$ to 20 and retrieved only the top-20 most relevant entities to strike a balance between the speed and effectiveness of subsequent retrieval and question-answering tasks.

**Multimodal-Fused Encoder Training and Inference.** The encoder model is initialized with pre-trained q-former weights using the LAVIS Library. We select the top 32 embeddings from the model output as our multi-modal fusion feature matrix, corresponding to the position and number of defined query tokens. Since InfoSeek training samples do not include labels for the evidence section, our encoder is trained on the E-VQA training set and then tested on both datasets. Given that each question-answer pair in the training samples corresponds to multiple query images, we use only the first query image to form the triplet with the question-answer pair as the actual training sample.

**Generator Training and Inference.** We primarily employ pre-trained LLM/MLLM models for ablation studies and most of the main experiments. To assess the performance of the generator fine-tuned under our retrieval system, we experiment with lightweight and efficient fine-tuning on LLaVA-1.5-7B and evaluate its VQA performance.

### 4.3 Main Results
The results of our method compared with other works are presented in Tables 2 and Table 3, primarily on entity retrieval and VQA performance.

**Retrieval Result.** Table 2 showcases the retrieval performance of various multimodal RAG approaches across two datasets. The "CLIP I-T" refers to the naive approach, where CLIP is used for the cross-modal similarity search, linking the query image to the wiki article with embeddings. By examining the Recall@1 results, our method (w. reranking) significantly outperforms other methods, which demonstrates the robust retrieval capability of our retrieval method. Additionally, the reranking in step two improves Recall@1 by 23.7% and 11.4% for the two datasets compared to step one alone, underscoring the effectiveness of multimodal retrieval and cross-step similarity propagation in boosting reranking performance. Furthermore, even without reranking, our method outperforms the full-scale retrieval strategies commonly used in other works across all retrieval ranges, demonstrating the significant advantage provided by the granularity alignment between the summary and the query image in step 1.

| Method | E-VQA R@1 | E-VQA R@5 | E-VQA R@10 | E-VQA R@20 | InfoSeek R@1 | InfoSeek R@5 | InfoSeek R@10 | InfoSeek R@20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CLIP I-T | 3.3 | 7.7 | 12.1 | 16.5 | 32.0 | 54.0 | 61.6 | 68.2 |
| Wiki-LLaVA | 3.3 | - | 9.9 | 13.2 | 36.9 | - | 66.1 | 71.9 |
| LLM-RA | - | - | - | - | 47.3 | 53.8 | - | - |
| mR2AG | - | - | - | - | 38.0 | - | 65.0 | 71.0 |
| ReflectiVA | 15.6 | 36.1 | - | 49.8 | 56.1 | 77.6 | - | 86.4 |
| EchoSight w/o. reranking | 13.3 | 31.3 | 41.0 | 48.8 | 45.6 | 67.1 | 73.0 | 77.9 |
| EchoSight w. reranking | 36.5 | 47.9 | 48.8 | 48.8 | 53.2 | 74.0 | 77.4 | 77.9 |
| **OMGM(ours) w/o. reranking** | 19.1 | 41.2 | 49.8 | 58.7 | 52.6 | 73.9 | 80.0 | 84.8 |
| **OMGM(ours) w. reranking** | **42.8** | **55.7** | **58.1** | **58.7** | **64.0** | **80.8** | **83.6** | **84.8** |

*Table 2: Retrieval results on the E-VQA test set and InfoSeek validation set. "w/o. reranking" and "w. reranking" represent the entity retrieval results after step one and step two, respectively. Best in bold, second-best underlined.*

**VQA Result.** Table 3 presents a comparison of our VQA results with the state-of-the-art methods. Our method achieves superior VQA results on both datasets compared to existing approaches, demonstrating the improved generation performance of downstream models after enhancing retrieval capabilities. Notably, our method, which only fine-tunes the retriever, outperforms most approaches that fine-tune downstream generators, highlighting its efficiency.

| Method | Generator Model | Gen. FT | Ret. FT | E-VQA | InfoSeek Unseen-Q | InfoSeek Unseen-E | InfoSeek Overall |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| RoRA-VLM | LLaVA-1.5-7B | ✓ | ✓ | 20.29 | 27.34 | 25.10 | - |
| Wiki-LLaVA | LLaVA-1.5-7B | ✓ | ✓ | 21.8 | 30.1 | 27.8 | 28.9 |
| LLM-RA | BLIP2-Flan-T5XL | ✓ | ✗ | - | 26.12 | 20.90 | 23.14 |
| EchoSight | Mistral-7B / LLaMA3-8B | ✗ | ✗ | 41.8 | - | 31.3 | - |
| mR2AG | LLaVA-1.5-7B | ✓ | ✓ | - | 40.6 | 39.8 | 40.2 |
| ReflectiVA | LLaVA-MORE-8B | ✓ | ✓ | 35.5 | 40.4 | 39.8 | 40.1 |
| **OMGM(ours)** | InternVL-2.5-8B | ✗ | ✗ | 48.72 | 37.16 | 35.1 | 36.1 |
| **OMGM(ours)** | LLaMA3-8B | ✗ | ✗ | 49.94 | 35.26 | 33.61 | 34.42 |
| **OMGM(ours)** | LLaVA-1.5-7B | ✗ | ✗ | **50.17** | **43.46** | **43.53** | **43.49** |

*Table 3: VQA accuracy comparison with the baselines. Gen. FT and Ret. FT indicate whether the generator and retriever of the method were fine-tuned, respectively (✓ = Yes, ✗ = No). Best in bold.*

### 4.4 Ablation Study
We conduct extensive experiments to assess the effectiveness of our framework and the design of each step, focusing on retrieval and question-answering performance.

**Impact of Retrieval Steps on VQA Results.** Table 4 records the VQA results on two datasets, as each step of our framework is executed in sequence. When only step one is completed, the generator receives the top-1 entity article as context. After step two is executed, the generator’s context is provided with the top-1 section. Each step in our multi-step framework progressively improves the VQA performance of the downstream generator, confirming the effectiveness of its design. Notably, step two offers the most significant enhancement, highlighting the powerful retrieval capability of our multimodal-fused reranker with similarity propagation.

| step-1 | step-2 | step-3 | E-VQA | InfoSeek |
| :---: | :---: | :---: | :--- | :--- |
| ✗ | ✗ | ✗ | 17.2 | 9.3 |
| ✓ | ✗ | ✗ | 25.45 | 18.87 |
| ✓ | ✓ | ✗ | 39.81 | 31.38 |
| ✓ | ✓ | ✓ | 41.81 | 33.29 |

*Table 4: The ablation study on the impact of different steps in our framework on the VQA results, using LLaVA-1.5-7B as the answer generator.*

**Impact of Retrieval Design on Initial Entity Search.** As the first step of multi-step retrieval, we perform a coarse-grained search for entity-related knowledge within a large knowledge base. We use the query image as the query, which uniquely contains entity information. Retrieval candidates can be selected based on different modalities and knowledge granularities. Table 5 presents our retrieval experiments using images, articles, titles, and summaries as candidates. The results indicate that the Image-to-Summary method achieves the best retrieval performance, suggesting that summaries offer better alignment with query images in terms of information granularity, leading to improved retrieval outcomes.

| Ret. Method | R@1 | R@5 | R@10 | R@20 |
| :--- | :--- | :--- | :--- | :--- |
| Image → Article | 13.2 | 27.7 | 35.5 | 41.7 |
| Image → Image | 13.4 | 31.8 | 41.9 | 48.8 |
| Image → Title | 17.5 | 31.9 | 38.6 | 44.8 |
| Image → Summary | 19.1 | 41.2 | 49.8 | 58.7 |

*Table 5: Ablation study on modalities and granularities design for entity retrieval in step 1.*

**Effect of Multimodal Fusion Reranking.** In step two, we rerank a small subset of entities obtained from the initial search to identify the most relevant one to the query. Consequently, we must leverage question-oriented, fine-grained textual information, such as the query question and the sections associated with each entity. In VQA tasks, both the query and the knowledge base are inherently multi-modal. Therefore, it is reasonable to consider incorporating multi-modal fusion features into the retrieval process. Based on this rationale, we experimented with four retrieval approaches: purely text-based retrieval, multimodal-to-text retrieval, text-to-multimodal retrieval, and fully multimodal retrieval. Using Q-Former as the base encoder model, we fine-tune all four retrieval modality approaches under the same training configuration and evaluated their performance. The results presented in Table 6 indicate that the method employing multimodal fusion features on both the query and candidate sides achieve the best retrieval performance.

| Ret. Modality | Sec. R@1 | R@1 | R@5 | R@10 |
| :--- | :--- | :--- | :--- | :--- |
| T → T | 24.6 | 30.7 | 51.8 | 57.4 |
| (I, T) → T | 22.5 | 28.7 | 51.3 | 57.0 |
| T → (I, T) | 24.3 | 30.3 | 51.1 | 56.9 |
| (I, T) → (I, T) | 32.8 | 40.2 | 54.8 | 57.8 |

*Table 6: Ablation study on retrieval modality of step two on E-VQA. "I" indicates image modality and "T" indicates text modality. "Sec. R@1" refers to the recall of the top-1 section. $\alpha$ is set to 0 for direct comparison.*

**Effect of Reranking Scope on Retrieval Performance.** The reranking scope $k$ influences both the number of entities filtered during the initial search and the range of multi-modal fused features extraction during reranking. Table 7 presents the retrieval results and the average retrieval time for the first two steps with varying $k$ values. Our framework can consistently improve the retrieval capacity while increasing $k$ from 10 to 100, accompanied with more retrieval time as well. To balance retrieval quality and efficiency, we set the reranking scope $k$ to 20 for experiments across both datasets.

| k | Sec. R@1 | R@1 | R@5 | R@10 | R@20 | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 31.6 | 39.0 | 48.8 | 49.8 | - | 0.630 |
| 20 | 34.7 | 42.8 | 55.7 | 58.1 | 58.7 | 1.110 |
| 50 | 37.4 | 45.9 | 61.3 | 64.6 | 66.7 | 2.420 |
| 100 | 38.8 | 47.4 | 65.0 | 69.4 | 72.2 | 4.642 |

*Table 7: The effect of the retrieval scope K on the retrieval results and time after step two on E-VQA.*

**Generalization on Additional Benchmark with Different Document Structures.** For E-VQA and InfoSeek documents, which consist of entity-based Wikipedia articles, we segmented them into sections based on their inherent structure. To evaluate the generalizability of OMGM on different segmentation settings, we tested our method on the OK-VQA dataset, which lacks a highly structured format and segmented using a rule-based segmentation method. As shown in Table 8, OMGM demonstrated strong performance, achieving higher retrieval accuracy (Pseudo Recall@5) and VQA scores compared to the strong baseline PreFLMR.

| Method | Pseudo Recall@5 | VQA score |
| :--- | :--- | :--- |
| PreFLMR | 70.9 | 61.88 |
| OMGM | 73.4 | 66.57 |

*Table 8: Comparison of the retrieval and VQA results of OMGM and PreFLMR on OK-VQA.*

**Quantitative evaluation on the efficiency of OMGM’s step-by-step approach.** To further perform the efficiency of the proposed OMGM framework, we conducted a comparative evaluation of OMGM, the one-step multimodal RAG method PreFLMR, and the direct use of LLaVA-1.5-7B for VQA on the E-VQA dataset. We evaluated three key metrics: average retrieval time, average inference time, and VQA performance.

| Method | Avg. Ret. Time | Avg. Inf. Time | VQA Result |
| :--- | :--- | :--- | :--- |
| LLaVA-1.5-7B | - | 1.432 | 17.00 |
| PreFLMR | 0.984 | 2.196 | 54.45 |
| OMGM | 0.402 | 2.023 | 63.39 |

*Table 9: Comparison of average retrieval and inference time as well as VQA performance on E-VQA.*

As shown in Table 9, OMGM achieves substantial improvements in VQA performance, surpassing both LLaVA-1.5-7B and PreFLMR. Despite employing a step-by-step retrieval strategy, OMGM maintains competitive inference efficiency. Additionally, when compared to the one-step PreFLMR, OMGM demonstrates a notable reduction in retrieval time, decreasing from 0.98s to 0.4s. This improvement is due to its orchestrated retrieval process, which is specifically designed to optimize the integration of different modalities and knowledge granularities at each step, achieving an effective balance between retrieval efficiency and performance.

## 5 Conclusion
In this paper, we propose a RAG system with multi-step multimodal retrieval. By employing queries and candidates of appropriate modalities at each step, the system aligns the information granularities for better retrieval. Our system capitalizes on cross-step similarity propagation to enhance retrieval interactions and employs a multimodal-fused design to fully exploit the rich multimodal information present in queries and candidates. Experimental results on mainstream KB-VQA datasets show that our approach surpasses existing approaches in retrieval performance. The comprehensive retrieval pipeline enables pre-trained models and lightly fine-tuned models to outperform the systems heavily reliant on extensive fine-tuning, and we reveal the rationality and effectiveness of it by ablation studies. These findings provide valuable insights for designing effective multimodal retrieval systems tailored to KB-VQA tasks.

## Limitations
Our RAG system focuses primarily on the design of the multimodal retrieval module. While it has demonstrated significant performance improvements, several limitations remain: 
1) Multimodal knowledge bases often include not only coarse-grained main images of entities but also numerous fine-grained secondary images linked to specific sections. Our current method does not exploit these secondary images to optimize multimodal retrieval, presenting a promising area for future exploration. 
2) Although our approach enhances VQA performance through improved retrieval, further exploration is needed to determine how generator models can more effectively utilize multimodal fusion features to enhance answer quality during the generation phase.

---

## Appendix A: Prompt Used in Our Methodology
In this section, we present the prompt templates utilized for invoking LLM/MLLM in our methodology, encompassing processes such as LLM-based summary generation and VQA across various datasets. 

**Table 15: Prompt template used to instruct the LLM to generate summaries offline for entity knowledge**
```text
System: You are a Wiki Summary Generator Assistant. Following is some information about you: 
## Profile 
- name: Wiki Summary Generator Assistant 
- language: English
- description: The Wiki Summary Generator Assistant is designed to create concise and informative summaries based on provided Wikipedia content. It extracts key aspects of the entity mentioned in the Wiki article, covering various dimensions such as history, characteristics, significance, appearance and impact.

## Workflows 
1. Input the provided Wikipedia content into the system.
2. Identify the main sections and key information related to the entity.
3. Synthesize this information into a well-structured summary.
4. Review and refine the summary for clarity, coherence, and completeness before finalizing.

## Rules
1. Focus on summarizing key details across multiple aspects (e.g., appearance, features, impact) of the entity.
2. Ensure the summary is concise, clear, and free of irrelevant details.
3. Retain the original meaning and context of the Wiki content while rephrasing it into a summary.

User: Following is the input Wikipedia content:
{ Wikipedia content}
Based on the above Wikipedia content, I would like you to generate a summary of the Wikipedia content.
Here is the summary of the Wikipedia content:
```

**Table 16: Prompt used for the VQA process of LLM/MLLM in E-VQA.**
```text
System: Answer the encyclopedic question about the given image. Don’t mention the visual content of image in your output. Directly output the answer of the question according to the context.
You are a helpful assistant for answering encyclopedic questions.
If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.

User:
{ Query Image} 
- Context: {Entity section} 
- Question: { Textual question}
The answer is:
```

**Table 17: Prompt used for the VQA process of LLM/MLLM in InfoSeek.**
```text
System: Answer the encyclopedic question about the given image. Don’t mention the visual content of image in your output. Directly output the answer of the question according to the context.
You are a helpful assistant for answering encyclopedic questions. Do not answer anything else.
If you need to answer questions about numbers or time, please output the corresponding numerical format directly. If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.

There is an example: 
- Context: # Wiki Article: Dolomites ## Section Title: Dolomites The Dolomites, also known as the Dolomite Mountains, Dolomite Alps or Dolomitic Alps, are a mountain range located in northeastern Italy. The Dolomites are located in the regions of Veneto, Trentino-Alto Adige/Südtirol and Friuli Venezia Giulia, covering an area shared between the provinces of Belluno, Vicenza, Verona, Trentino, South Tyrol, Udine and Pordenone.
- Question: Which city or region does this mountain locate in?
Just answer the questions, no explanations needed. Short answer is: Province of Belluno 

User:
{ Query Image} 
- Context: {Entity section} 
- Question: { Textual question}
Just answer the questions, no explanations needed. Short answer is:
```
