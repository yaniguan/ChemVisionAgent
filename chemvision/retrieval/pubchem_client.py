"""PubChem REST client — no API key required.

Uses the PubChem PUG-REST API (https://pubchem.ncbi.nlm.nih.gov/rest/pug/).
All calls are synchronous and cached in-memory to avoid hammering the service.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PROPS = (
    "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,"
    "IUPACName,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,"
    "RotatableBondCount,HeavyAtomCount,Complexity"
)


class PubChemClient:
    """Thin wrapper around the PubChem PUG-REST API.

    All results are cached per-session so repeated calls for the same
    compound don't hit the network twice.

    Example
    -------
    >>> client = PubChemClient()
    >>> props = client.fetch_by_smiles("CC(=O)Oc1ccccc1C(=O)O")
    >>> props["IUPACName"]
    '2-acetyloxybenzoic acid'
    """

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "ChemVisionAgent/0.2 (research)"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def fetch_by_smiles(self, smiles: str) -> dict[str, Any]:
        """Return PubChem property dict for the given SMILES.

        Falls back to an empty dict on any network / parse error so
        callers never have to guard for exceptions.
        """
        return self._fetch_props("smiles", smiles)

    def fetch_by_name(self, name: str) -> dict[str, Any]:
        """Return PubChem property dict for a compound name (e.g. 'aspirin')."""
        return self._fetch_props("name", name)

    def fetch_by_cid(self, cid: int) -> dict[str, Any]:
        """Return PubChem property dict for a numeric CID."""
        return self._fetch_props("cid", str(cid))

    def get_similar_compounds(
        self, smiles: str, threshold: int = 90, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Return up to *max_results* compounds with Tanimoto ≥ threshold.

        Parameters
        ----------
        smiles:
            Query molecule as SMILES.
        threshold:
            Tanimoto similarity threshold (0–100).
        max_results:
            Maximum number of similar CIDs to fetch properties for.
        """
        cids = self._similar_cids(smiles, threshold, max_results)
        results: list[dict[str, Any]] = []
        for cid in cids:
            props = self.fetch_by_cid(cid)
            if props:
                results.append(props)
            time.sleep(0.1)  # be polite to PubChem
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=256)
    def _fetch_props(self, id_type: str, identifier: str) -> dict[str, Any]:
        url = f"{_BASE}/compound/{id_type}/{requests.utils.quote(identifier)}/property/{_PROPS}/JSON"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            table = resp.json()["PropertyTable"]["Properties"]
            return dict(table[0]) if table else {}
        except (requests.RequestException, TimeoutError) as exc:
            logger.error("PubChem request failed for %s/%s: %s", id_type, identifier, exc)
            return {}
        except (KeyError, ValueError) as exc:
            logger.warning("PubChem response parse error for %s/%s: %s", id_type, identifier, exc)
            return {}

    def _similar_cids(self, smiles: str, threshold: int, max_results: int) -> list[int]:
        url = (
            f"{_BASE}/compound/fastsimilarity_2d/smiles/"
            f"{requests.utils.quote(smiles)}/cids/JSON"
            f"?Threshold={threshold}&MaxRecords={max_results}"
        )
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["IdentifierList"]["CID"][:max_results]
        except (requests.RequestException, TimeoutError) as exc:
            logger.error("PubChem similarity search failed for %r: %s", smiles, exc)
            return []
        except (KeyError, ValueError) as exc:
            logger.warning("PubChem similarity response parse error: %s", exc)
            return []
