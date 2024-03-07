# conftest.py
# pour éviter que pytest nsoit perdu
# à caudse des adresses relatives

import pytest
import os

@pytest.fixture(autouse=True)
def change_working_directory():
    # Change to the root directory where your tests are located
    os.chdir('/home/ubuntu/Bureau/OC/Projet5_oudelet_kevin/5_6_API/')

    # The fixture will revert the working directory back to the original after the test
    yield

    # Optionally, you can change back to the original working directory after the test
    # os.chdir('/home/ubuntu/Bureau/OC/Projet5_oudelet_kevin/')
