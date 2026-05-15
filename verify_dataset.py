from ho3d_dataset import HO3DDataset, discover_sequences, split_sequences, group_sequences_by_subject, HELDOUT_SEQUENCES_PER_SUBJECT
from paths import DATASET_ROOT

all_sequences = discover_sequences(DATASET_ROOT)
grouped = group_sequences_by_subject(all_sequences)
train_seqs, val_seqs = split_sequences(all_sequences, HELDOUT_SEQUENCES_PER_SUBJECT)

print("Per-subject sequence counts:")
for subject, sequences in sorted(grouped.items()):
    print(f"  {subject}: {len(sequences)} ({sequences})")

print(f"\nTrain sequences ({len(train_seqs)}): {train_seqs}")
print(f"Val sequences ({len(val_seqs)}): {val_seqs}")
print()

dataset = HO3DDataset("train")
sample = dataset[0]

print(f"train samples: {len(dataset)}")
print(f"first sample keys: {list(sample.keys())}")
print(f"edge map shape: {sample['edge_map'].shape}")
print(f"theta shape: {sample['theta'].shape}")
print(f"beta shape: {sample['beta'].shape}")
print(f"joints_3d shape: {sample['joints_3d'].shape}")
print(f"sequence: {sample['sequence']}, frame: {sample['frame_id']}")

val_dataset = HO3DDataset("val")
print(f"\nval samples: {len(val_dataset)}")
