from __future__ import annotations

from smash._constant import NET_OPTIMIZER, LAYER_NAME, ACTIVATION_FUNCTION

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple, Numeric


def _standardize_add_layer(layer: str) -> str:
    if isinstance(layer, str):
        layer_standardized = layer.lower().capitalize()

        if layer_standardized in LAYER_NAME:
            layer = layer_standardized

        else:
            raise ValueError(f"Unknown layer type '{layer}'. Choices: {LAYER_NAME}")

    else:
        raise TypeError(f"layer argument must be str")
    
    return layer

def _standardize_add_options(layer: str, options: dict) -> dict:
    if isinstance(options, dict):
        if layer == "Activation":
            try:
                option_name = options["name"]

            except KeyError:
                raise ValueError(
                    f"Key 'name' in options argument must be specified for the activation function"
                )

            if isinstance(option_name, str):
                activation_function = [name.lower() for name in ACTIVATION_FUNCTION]

                try:
                    ind = activation_function.index(option_name.lower())

                except ValueError:
                    raise ValueError(
                        f"Unknown activation function '{option_name}'. Choices: {ACTIVATION_FUNCTION}"
                    )

                options["name"] = ACTIVATION_FUNCTION[ind]

            else:
                raise TypeError(f"Key 'name' in options argument must be a str")

        else:
            pass

    else:
        raise TypeError(f"options argument must be a dictionary")

    return options


def _standardize_add_args(layer: str, options: dict) -> AnyTuple:
    layer = _standardize_add_layer(layer)

    options = _standardize_add_options(layer, options)

    return (layer, options)


def _standardize_compile_optimizer(optimizer: str) -> str:
    if isinstance(optimizer, str):
        net_optimizer = [name.lower() for name in NET_OPTIMIZER]

        try:
            ind = net_optimizer.index(optimizer.lower())

        except ValueError:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Choices: {NET_OPTIMIZER}"
            )

        optimizer = NET_OPTIMIZER[ind]

    else:
        raise TypeError(f"optimizer argument must be str")

    return optimizer


def _standardize_compile_random_state(random_state: Numeric | None) -> int:
    if random_state is None:
        pass

    else:
        if not isinstance(random_state, (int, float)):
            raise TypeError(
                "random_state argument must be of Numeric type (int, float)"
            )

        random_state = int(random_state)

        if random_state < 0 or random_state > 4_294_967_295:
            raise ValueError("random_state argument must be between 0 and 2**32 - 1")

    return random_state


def _standadize_compile_options(options: dict | None) -> dict:
    if options is None:
        options = {}

    elif isinstance(options, dict):
        pass

    else:
        raise TypeError(f"options argument must be a dictionary or None")

    return options


def _standardize_compile_args(
    optimizer: str, random_state: Numeric | None, options: dict | None
) -> AnyTuple:
    optimizer = _standardize_compile_optimizer(optimizer)

    random_state = _standardize_compile_random_state(random_state)

    options = _standadize_compile_options(options)

    return (optimizer, random_state, options)
