from audio_utils import select_audio_device


class FakeDeviceInfo(dict):
    pass


class FakeSoundDevice:
    def __init__(self, devices, default_device=1):
        self._devices = devices
        self.default = type("Default", (), {"device": default_device})()

    def query_devices(self, device=None, kind=None):
        if device is None:
            return self._devices
        if isinstance(device, int):
            return self._devices[device]
        return self._devices[device]


def test_select_audio_device_prefers_default_output():
    devices = [
        FakeDeviceInfo(index=0, max_output_channels=0),
        FakeDeviceInfo(index=1, max_output_channels=2, default_samplerate=48000),
    ]
    sd = FakeSoundDevice(devices, default_device=1)

    assert select_audio_device(sd, kind="output") == 1


def test_select_audio_device_falls_back_to_first_available():
    devices = [
        FakeDeviceInfo(index=0, max_output_channels=2, default_samplerate=44100),
    ]
    sd = FakeSoundDevice(devices, default_device=99)

    assert select_audio_device(sd, kind="output") == 0
