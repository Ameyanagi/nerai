"""Validated parameter-wise box bounds."""

from std.io import Writable, Writer


struct Bounds(Copyable, Equatable, Writable):
    """Lower and upper bounds for a least-squares parameter vector.

    Construction establishes every stored invariant. Trusted accessors do not
    revalidate; direct mutation of underscore-prefixed storage is out of
    contract, and ``validate()`` provides an explicit checkpoint afterward.

    Convention: Solver accepted and trial iterates stay strictly inside every
    finite bound. ``contains()`` answers the separate closed-interval question.
    Returned active bounds use ``xtol`` as their relative tolerance when it is
    enabled and ``1.5e-8`` otherwise; finite lower bounds win a tolerance tie.
    An outward-active coordinate is held at that tolerance while tangent
    coordinates finish converging.
    """

    var _lower: List[Float64]
    var _upper: List[Float64]

    def __init__(out self, lower: List[Float64], upper: List[Float64]) raises:
        """Copy and validate one lower and upper endpoint per parameter."""
        self._lower = lower.copy()
        self._upper = upper.copy()
        self.validate()

    def validate(self) raises:
        """Revalidate all endpoints after unusual direct mutation."""
        if len(self._lower) != len(self._upper):
            raise Error(
                String(
                    "bounds have ",
                    len(self._lower),
                    " lower entries and ",
                    len(self._upper),
                    " upper entries",
                )
            )
        if len(self._lower) < 1:
            raise Error("bounds require at least one parameter")

        for index in range(len(self._lower)):
            var lower = self._lower[index]
            var upper = self._upper[index]
            if lower != lower:
                raise Error(
                    String("bounds[", index, "] lower endpoint is NaN: ", lower)
                )
            if upper != upper:
                raise Error(
                    String("bounds[", index, "] upper endpoint is NaN: ", upper)
                )
            if lower >= upper:
                raise Error(
                    String(
                        "bounds[",
                        index,
                        "] are empty: lower ",
                        lower,
                        " must be strictly below upper ",
                        upper,
                    )
                )

    def lower(self, index: Int) -> Float64:
        """Return one trusted lower endpoint."""
        return self._lower[index]

    def upper(self, index: Int) -> Float64:
        """Return one trusted upper endpoint."""
        return self._upper[index]

    def contains(self, parameters: List[Float64]) -> Bool:
        """Return whether a vector is in the closed box, including endpoints.

        This closed-interval query differs deliberately from the solver's
        strict-interior iterate convention.
        """
        if len(parameters) != self.parameter_count():
            return False
        for index in range(len(parameters)):
            if not (
                self._lower[index] <= parameters[index]
                and parameters[index] <= self._upper[index]
            ):
                return False
        return True

    def parameter_count(self) -> Int:
        """Return the number of bounded parameters."""
        return len(self._lower)

    def __eq__(self, other: Self) -> Bool:
        """Return whether every endpoint is exactly equal."""
        if self.parameter_count() != other.parameter_count():
            return False
        for index in range(self.parameter_count()):
            if (
                self._lower[index] != other._lower[index]
                or self._upper[index] != other._upper[index]
            ):
                return False
        return True

    def __str__(self) -> String:
        """Return the stable multiline bounds block."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write one closed interval per parameter with a trailing newline."""
        for index in range(self.parameter_count()):
            var label = String("bounds[", index, "]")
            writer.write(label)
            for _ in range(label.byte_length(), 22):
                writer.write(" ")
            writer.write("[", self._lower[index], ", ", self._upper[index], "]\n")
