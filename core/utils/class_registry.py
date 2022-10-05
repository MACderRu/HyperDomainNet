import inspect
import typing
import omegaconf
import dataclasses
import typing as tp


class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def make_dataclass_from_func(self, func, name, arg_keys):
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                dataclasses.field(default=v.default),
            )
            for k, v in args.items()
        ]
        args = [
            arg
            for arg in args
            if (arg[0] != "self" and arg[0] != "args" and arg[0] != "kwargs")
        ]
        if arg_keys:
            self.arg_keys = arg_keys
            arg_classes = dict()
            for key in arg_keys:
                arg_classes[key] = dataclasses.make_dataclass(key, args)
            return dataclasses.make_dataclass(
                name,
                [
                    (k, v, dataclasses.field(default=v()))
                    for k, v in arg_classes.items()
                ],
            )
        return dataclasses.make_dataclass(name, args)

    def make_dataclass_from_classes(self):
        return dataclasses.make_dataclass(
            'Name',
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.classes.items()
            ],
        )

    def make_dataclass_from_args(self):
        return dataclasses.make_dataclass(
            'Name',
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.args.items()
            ],
        )

    def _add_single_obj(self, obj, name, arg_keys):
        self.classes[name] = obj
        if inspect.isfunction(obj):
            self.args[name] = self.make_dataclass_from_func(
                obj, name, arg_keys
            )
        elif inspect.isclass(obj):
            self.args[name] = self.make_dataclass_from_func(
                obj.__init__, name, arg_keys
            )

    def add_to_registry(self, names: tp.Union[str, tp.List[str]], arg_keys=None):
        if not isinstance(names, list):
            names = [names]

        def decorator(obj):
            for name in names:
                self._add_single_obj(obj, name, arg_keys)

            return obj
        return decorator

    def __contains__(self, name: str):
        return name in self.args.keys()

    def __repr__(self):
        return f"{list(self.args.keys())}"
