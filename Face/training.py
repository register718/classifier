from datasets.dataset_imdb import SiameseIMDBDataset as SiameseDataset

sets = SiameseDataset("/media/hitchcock/data21/data/imdb")

print(sets[0])