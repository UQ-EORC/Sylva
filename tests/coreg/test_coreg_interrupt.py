# The coregistration reports through a Python callable; an exception that
# callable raises (KeyboardInterrupt included) ends the run rather than
# waiting for it to finish.
import pytest

from sylva.coreg import CoregConfig, coregister


@pytest.fixture(scope="module")
def quiet_config():
    cfg = CoregConfig()
    cfg.verbose = False
    return cfg


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_an_exception_in_the_progress_callable_is_raised_and_it_is_not_called_again(small_survey, quiet_config, error):
    seen = []

    def progress(message):
        seen.append(message)
        raise error("stop")

    with pytest.raises(error, match="stop"):
        coregister(small_survey.clouds, quiet_config, progress=progress)
    assert len(seen) == 1


def test_a_quiet_progress_callable_sees_the_whole_run(small_survey, quiet_config):
    seen = []
    result = coregister(small_survey.clouds, quiet_config, progress=seen.append)
    assert len(seen) > 1
    assert len(result.poses) == len(small_survey.clouds)
