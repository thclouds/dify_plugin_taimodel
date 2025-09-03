from typing import Mapping

from dify_plugin.entities.model import AIModelEntity, I18nObject

from dify_plugin.interfaces.model.openai_compatible.text_embedding import OAICompatEmbeddingModel


class OpenAITextEmbeddingModel(OAICompatEmbeddingModel):

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(
                en_US=credentials["display_name"], zh_Hans=credentials["display_name"]
            )

        return entity
