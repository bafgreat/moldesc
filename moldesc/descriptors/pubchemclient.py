from __future__ import print_function
import re
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

import requests
import pubchempy as pcp
from pubchempy import get_assays


class PubChemClient:
    """
    Client for retrieving compound and bioassay information from the PubChem database.

    This class provides a lightweight interface to the PubChem REST services via
    the :mod:`pubchempy` package.
    The client supports:

    - Retrieval of basic compound information from a chemical identifier.
    - Retrieval of PubChem BioAssay (AID) metadata.
    - In-memory caching of assay information to reduce repeated network requests.
    - Automatic association of compounds with their available bioassays.

    Parameters
    ----------
    max_assays : int, optional
        Maximum number of PubChem BioAssays to retrieve for a single compound.
        Default is 50.

    Attributes
    ----------
    max_assays : int
        Maximum number of assays retrieved per compound.

    assay_cache : dict
        Dictionary used to cache previously downloaded assay metadata,
        avoiding repeated requests for the same AID.
    """

    def __init__(self, max_assays=50):
        self.max_assays = max_assays
        self.assay_cache = {}

    def get_ligand(self, identifier, namespace="smiles"):
        """
        Retrieve basic compound information from PubChem.

        Searches PubChem using the supplied chemical identifier and returns
        a dictionary containing the primary compound annotations required by
        the Ligand Atlas.

        Parameters
        ----------
        identifier : str
            Chemical identifier used for the PubChem search. Examples include
            SMILES strings, InChIKeys, compound names, or PubChem CIDs.

        namespace : str, optional
            Type of identifier supplied. This argument is passed directly to
            :func:`pubchempy.get_compounds`.

            Common values include:

            - ``"smiles"``
            - ``"name"``
            - ``"inchikey"``
            - ``"cid"``
            - ``"inchi"``

            Default is ``"smiles"``.

        Returns
        -------
        dict or None
            Dictionary containing:

            - ``cid`` : PubChem Compound ID.
            - ``iupac_name`` : Preferred IUPAC name.
            - ``aids`` : List of PubChem BioAssay IDs.
            - ``synonyms`` : List of common chemical names.

            Returns ``None`` if no matching compound is found.
        """
        compounds = pcp.get_compounds(identifier, namespace)
        if not compounds:
            return None

        c = compounds[0]

        return {
            "cid": c.cid or None,
            "iupac_name": c.iupac_name or None,
            "aids": c.aids or [],
            # "sids": c.sids or [],
            "synonyms": c.synonyms if c.synonyms else [],
        }

    def get_assay(self, aid):
        """
        Retrieve metadata for a PubChem BioAssay.

        Downloads the metadata describing a PubChem BioAssay (AID). Retrieved
        assays are cached in memory so that repeated requests for the same AID
        do not require additional network calls.

        Parameters
        ----------
        aid : int
            PubChem BioAssay identifier (AID).

        Returns
        -------
        dict or None
            Dictionary containing:

            - ``aid`` : BioAssay identifier.
            - ``name`` : Assay title.
            - ``description`` : Description of the assay protocol.
            - ``target`` : Biological target, if available.
            - ``results`` : List of assay endpoints (e.g., IC50, GI50, EC50).

            Returns ``None`` if the assay cannot be retrieved.
        """
        if aid in self.assay_cache:
            return self.assay_cache[aid]

        assays = get_assays(aid)
        if not assays:
            return None

        assay = assays[0].to_dict()

        data = {
            "aid": assay.get("aid"),
            "name": assay.get("name"),
            "description": assay.get("description", []),
            "target": assay.get("target"),
            "results": assay.get("results", []),
        }

        self.assay_cache[aid] = data
        return data

    def get_ligand_with_assays(self, identifier, namespace="smiles"):
        """
        Retrieve a compound together with its associated PubChem BioAssays.

        This is a convenience method that first retrieves the compound metadata
        using :meth:`get_ligand`, then downloads metadata for each associated
        PubChem BioAssay.

        The number of downloaded assays is limited by ``max_assays``.

        Parameters
        ----------
        identifier : str
            Chemical identifier used to search PubChem.

        namespace : str, optional
            Namespace describing the identifier type.
            Default is ``"smiles"``.

        Returns
        -------
        dict or None
            Dictionary containing the ligand information returned by
            :meth:`get_ligand`, with an additional key:

            ``assays``
                List containing the metadata for each retrieved PubChem BioAssay.

            Returns ``None`` if the compound cannot be found.
        """
        ligand = self.get_ligand(identifier, namespace)

        if ligand is None:
            return None

        aids = ligand.get("aids", [])[:self.max_assays]

        ligand["assays"] = []

        for aid in aids:
            assay = self.get_assay(aid)
            if assay is not None:
                ligand["assays"].append(assay)

        return ligand