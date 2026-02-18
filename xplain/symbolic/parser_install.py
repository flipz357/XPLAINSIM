import logging
import os

logger = logging.getLogger(__name__)

_AMRLIB_SMATCHPP_INSTALL_MESSAGE = (
    "AMR functionality requires optional dependencies.\n"
    "Install via:\n\n"
    "    pip install amrlib smatchpp\n"
)

def install_default_amr_model():
    import os
    import subprocess
    import urllib.request
    import amrlib
    url = "https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz"
    logger.info("Downloading default AMR parser model...")

    amrlibpath = os.path.dirname(amrlib.__file__)
    datapath = os.path.join(amrlibpath, "data")
    os.makedirs(datapath, exist_ok=True)

    archive_path = os.path.join(datapath, "model.tar.gz")

    urllib.request.urlretrieve(url, archive_path)
    subprocess.check_call(["tar", "-xvzf", archive_path, "-C", datapath])
    subprocess.check_call(["mv",
                           os.path.join(datapath, "model_parse_xfm_bart_base-v0_1_0"),
                           os.path.join(datapath, "model_stog")])

    logger.info("AMR parser model installed successfully.")

