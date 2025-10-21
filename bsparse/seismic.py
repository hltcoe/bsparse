# TODO implement
from seismic import SeismicIndex


json_input_file = "combined.jsonl"

# batched_indexing is important to speed seismic up a lot
index = SeismicIndex.build(json_input_file, batched_indexing=100000)
print("Number of documents:", index.len)
print("Avg number of non-zero components:", index.nnz / index.len)
print("Dimensionality of the vectors:", index.dim)

index.print_space_usage_byte()

index.save("SEISMICINDEX")
