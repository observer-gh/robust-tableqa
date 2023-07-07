## Robust Table Question Answering

Code for training and evaluation the transformer-based robust Table QA models introduced in the following ACL 2023 papers:

<div align="center">    
 
### An <u>I</u>nner <u>T</u>able <u>R</u>etriever for robust table question answering

[![Paper](https://img.shields.io/badge/Paper-ACL_Proceedings-red)](https://www.amazon.science/publications/an-inner-table-retriever-for-robust-table-question-answering)
[![Conference](https://img.shields.io/badge/Conference-ACL--2023-blue)](https://2023.aclweb.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

Inner Table Retriever (ITR) is a general-purpose approach for handling long tables in TableQA 
that extracts sub-tables to preserve the most relevant information for a question. 
ITR can be easily integrated into existing systems to improve their accuracy achieve state-of-the-art results.

If you find our paper, code or framework useful, please put a link to this repo and reference this work in your paper:

```
@Inproceedings{Lin2023,
 author = {Weizhe Lin and Rexhina Blloshmi  and Bill Byrne and Adrià de Gispert and Gonzalo Iglesias},
 title = {An inner table retriever for robust table question answering},
 year = {2023},
 url = {https://www.amazon.science/publications/an-inner-table-retriever-for-robust-table-question-answering},
 booktitle = {ACL 2023},
}
```

For more info and details on how to install/run check [COMING SOON](). 

<hr> 

<div align="center">
 
### LI-RAGE: <u>L</u>ate <u>I</u>nteraction <u>R</u>etrieval <u>A</u>ugmented <u>G</u>eneration with <u>E</u>xplicit signals for open-domain table question answering

[![Paper](https://img.shields.io/badge/Paper-Amazon_Science-orange)](https://www.amazon.science/publications/li-rage-late-interaction-retrieval-augmented-generation-with-explicit-signals-for-open-domain-table-question-answering)
[![Conference](https://img.shields.io/badge/Conference-ACL_2023-red)](https://2023.aclweb.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>
LI-RAGE is a framework for open-domain TableQA which addresses several limitations thanks to: 

1) applying late interaction models which enforce a finer-grained interaction between question and table embeddings at retrieval time. 
2) incorporating a joint training scheme of the retriever and reader with explicit table-level signals, and 
3) embedding a binary relevance token as a prefix to the answer generated by the reader, 
so we can determine at inference time whether the table used to answer the question is reliable and filter accordingly. 
The combined strategies set a new state-to-the-art performance on two public open-domain TableQA datasets.



If you find our paper, code or framework useful, please put a link to this repo and reference this work in your paper:

```
@Inproceedings{Lin2023,
 author = {Weizhe Lin and Rexhina Blloshmi and Bill Byrne and Adrià de Gispert and Gonzalo Iglesias},
 title = {LI-RAGE: Late interaction retrieval augmented generation with explicit signals for open-domain table question answering},
 year = {2023},
 url = {https://www.amazon.science/publications/li-rage-late-interaction-retrieval-augmented-generation-with-explicit-signals-for-open-domain-table-question-answering},
 booktitle = {ACL 2023},
}
```

For more info and details on how to install/run check [COMING SOON](). 

<hr>


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the LICENSE file.

The sample code within this documentation is made available under the MIT-0 license. See the LICENSE-SAMPLECODE file.
