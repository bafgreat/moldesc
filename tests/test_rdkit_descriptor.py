from moldesc.descriptors.rdkitdescriptors import RDKitDescriptors

rdkit_data = RDKitDescriptors(smile="C=CCSC1=NN=C(S1)SCC=C")
data = rdkit_data.to_dict()

assert len(data) == 7