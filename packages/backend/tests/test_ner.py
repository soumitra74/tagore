import pytest

from app.services.ner import ENTITY_LABELS, NERService, normalize_entity_text


class FakeGlinerModel:
    def __init__(self, results):
        self._results = results
        self.received_args = None

    def predict_entities(self, text, labels, threshold=0.5):
        self.received_args = (text, labels, threshold)
        return self._results


class FakeNERService(NERService):
    def __init__(self, fake_model):
        super().__init__()
        self._fake_model = fake_model

    def _get_model(self):
        return self._fake_model


def test_normalize_entity_text_trims_lowercases_and_collapses_whitespace():
    assert normalize_entity_text("  TM-104598  ") == "tm-104598"
    assert normalize_entity_text("Data   Counter") == "data counter"


@pytest.mark.asyncio
async def test_extract_maps_gliner_output_to_extracted_entities():
    fake_model = FakeGlinerModel([
        {"start": 14, "end": 18, "text": "9017", "label": "DATA_COUNTER_ID", "score": 0.91},
    ])
    service = FakeNERService(fake_model)

    result = await service.extract("For Data Counter 9017, what were...")

    assert len(result) == 1
    entity = result[0]
    assert entity.entity_type == "DATA_COUNTER_ID"
    assert entity.entity_text == "9017"
    assert entity.normalized_text == "9017"
    assert entity.char_start == 14
    assert entity.char_end == 18
    assert fake_model.received_args[0] == "For Data Counter 9017, what were..."
    assert fake_model.received_args[1] == ENTITY_LABELS


@pytest.mark.asyncio
async def test_extract_returns_empty_list_for_no_entities():
    service = FakeNERService(FakeGlinerModel([]))
    result = await service.extract("no entities in this sentence")
    assert result == []


@pytest.mark.asyncio
async def test_warm_up_does_not_raise():
    service = FakeNERService(FakeGlinerModel([]))
    await service.warm_up()
