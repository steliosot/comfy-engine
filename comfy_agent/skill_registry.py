import importlib
import os


class SkillRegistry:

    _skills = {}

    @classmethod
    def register(cls, name, func):
        cls._skills[name] = func

    @classmethod
    def get(cls, name):
        return cls._skills[name]

    @classmethod
    def load_skills(cls, base_path="skills"):

        for folder in os.listdir(base_path):

            path = os.path.join(base_path, folder)

            if os.path.isdir(path):

                try:
                    module = importlib.import_module(
                        f"skills.{folder}.skill"
                    )

                    cls.register(folder, module.run)

                except Exception:
                    pass