"""DECIMALs of more than 38 digits, in and out of every Python form.

Out of the engine a wide DECIMAL is a `decimal.Decimal` in rows, a float64 in NumPy and pandas like every narrower
DECIMAL, and a decimal256 in Arrow up to 76 digits. Into the engine every finite `decimal.Decimal` is an exact DECIMAL
of the width it needs.
"""

import math
from decimal import Decimal, localcontext

import pytest

import duckdb

# (width, scale) of the types under test: just past the fixed-size tiers, the widest Arrow decimal and one past it,
# the widest width a single byte holds, and one beyond that
SHAPES = [(39, 0), (39, 5), (76, 10), (77, 3), (255, 20), (300, 37)]


def exactly(compute):
    """The result of Decimal arithmetic, computed with enough digits to be exact."""
    with localcontext() as context:
        context.prec = 1000
        return compute()


def decimal_of(width: int, scale: int, *, negative: bool = False) -> Decimal:
    """A value using every digit of DECIMAL(width, scale)."""
    digits = "".join(str(i % 9 + 1) for i in range(width))
    text = digits[: width - scale] + ("." + digits[width - scale :] if scale else "")
    return Decimal(("-" if negative else "") + text)


@pytest.fixture
def con():
    with duckdb.connect() as connection:
        yield connection


@pytest.fixture
def arrow_con(con):
    # decimal256 is in the Arrow format from version 1.1 on
    con.execute("SET arrow_output_version = '1.1'")
    return con


class TestRows:
    @pytest.mark.parametrize(("width", "scale"), SHAPES)
    def test_fetch_is_exact(self, con, width, scale):
        positive = decimal_of(width, scale)
        negative = decimal_of(width, scale, negative=True)
        rel = con.sql(
            f"SELECT v::DECIMAL({width},{scale}) AS v FROM (VALUES ('{positive}'), (NULL), ('{negative}')) t(v)"
        )
        assert str(rel.types[0]) == f"DECIMAL({width},{scale})"
        assert rel.fetchall() == [(positive,), (None,), (negative,)]
        cursor = con.execute(rel.sql_query())
        assert cursor.fetchone() == (positive,)
        assert cursor.fetchmany(2) == [(None,), (negative,)]

    def test_default_session_arithmetic(self, con):
        # The sum of two DECIMAL(38,2) is a DECIMAL(39,2) in a session with default settings
        value = Decimal("9" * 36 + ".99")
        rel = con.sql(f"SELECT a + a AS s FROM (SELECT '{value}'::DECIMAL(38,2) AS a)")
        assert str(rel.types[0]) == "DECIMAL(39,2)"
        assert rel.fetchall() == [(exactly(lambda: value * 2),)]

    def test_nested(self, con):
        value = decimal_of(50, 5, negative=True)
        rows = con.sql(
            f"SELECT ['{value}'::DECIMAL(50,5), NULL] AS l, {{'d': '{value}'::DECIMAL(50,5)}} AS s"
        ).fetchall()
        assert rows == [([value, None], {"d": value})]


class TestNumpyAndPandas:
    """A DECIMAL is a float64 in NumPy and pandas at every width, a NULL is masked."""

    @pytest.mark.parametrize(("width", "scale"), SHAPES)
    def test_fetchnumpy(self, con, width, scale):
        np = pytest.importorskip("numpy")
        positive = decimal_of(width, scale)
        negative = decimal_of(width, scale, negative=True)
        result = con.sql(
            f"SELECT v::DECIMAL({width},{scale}) AS v FROM (VALUES ('{positive}'), (NULL), ('{negative}')) t(v)"
        ).fetchnumpy()["v"]
        assert result.dtype == np.float64
        assert list(np.ma.getmaskarray(result)) == [False, True, False]
        # rounded once, to the nearest double
        assert result[0] == float(positive)
        assert result[2] == float(negative)

    @pytest.mark.parametrize(("width", "scale"), SHAPES)
    def test_df(self, con, width, scale):
        pytest.importorskip("pandas")
        positive = decimal_of(width, scale)
        negative = decimal_of(width, scale, negative=True)
        frame = con.sql(
            f"SELECT v::DECIMAL({width},{scale}) AS v FROM (VALUES ('{positive}'), (NULL), ('{negative}')) t(v)"
        ).df()
        assert str(frame["v"].dtype) == "float64"
        assert frame["v"][0] == float(positive)
        assert math.isnan(frame["v"][1])
        assert frame["v"][2] == float(negative)

    def test_same_as_a_narrow_decimal(self, con):
        pd = pytest.importorskip("pandas")
        np = pytest.importorskip("numpy")
        # More rows than one vector holds, a NULL every seventh row
        query = """
            SELECT v::DECIMAL(38,3) AS narrow, v::DECIMAL(60,3) AS wide
            FROM (SELECT CASE WHEN i % 7 = 0 THEN NULL ELSE (i - 2500) * 1.125 END AS v FROM range(5000) t(i))
        """
        frame = con.sql(query).df()
        pd.testing.assert_series_equal(frame["narrow"], frame["wide"], check_names=False)
        arrays = con.sql(query).fetchnumpy()
        assert type(arrays["wide"]) is type(arrays["narrow"])
        assert arrays["wide"].dtype == arrays["narrow"].dtype
        np.testing.assert_array_equal(np.ma.getmaskarray(arrays["wide"]), np.ma.getmaskarray(arrays["narrow"]))
        np.testing.assert_array_equal(arrays["wide"].filled(0), arrays["narrow"].filled(0))

    def test_without_nulls(self, con):
        pytest.importorskip("pandas")
        frame = con.sql("SELECT (i * 1.5)::DECIMAL(50,2) AS v FROM range(3) t(i)").df()
        assert str(frame["v"].dtype) == "float64"
        assert list(frame["v"]) == [0.0, 1.5, 3.0]

    def test_beyond_the_range_of_a_double(self, con):
        np = pytest.importorskip("numpy")
        result = con.sql(
            "SELECT v::DECIMAL(400,0) AS v FROM (VALUES ('1' || repeat('0', 399)), ('-1' || repeat('0', 399))) t(v)"
        ).fetchnumpy()["v"]
        assert result.dtype == np.float64
        assert list(result) == [math.inf, -math.inf]


class TestArrowOutput:
    @pytest.mark.parametrize(("width", "scale"), [(39, 0), (39, 5), (76, 10)])
    def test_decimal256(self, arrow_con, width, scale):
        pa = pytest.importorskip("pyarrow")
        positive = decimal_of(width, scale)
        negative = decimal_of(width, scale, negative=True)
        rel = arrow_con.sql(
            f"SELECT v::DECIMAL({width},{scale}) AS v FROM (VALUES ('{positive}'), (NULL), ('{negative}')) t(v)"
        )
        table = rel.to_arrow_table()
        assert table.schema.field("v").type == pa.decimal256(width, scale)
        assert table.column("v").to_pylist() == [positive, None, negative]
        reader = rel.to_arrow_reader()
        assert reader.schema.field("v").type == pa.decimal256(width, scale)
        assert reader.read_all().column("v").to_pylist() == [positive, None, negative]

    def test_default_session_arithmetic(self, arrow_con):
        pa = pytest.importorskip("pyarrow")
        value = Decimal("9" * 36 + ".99")
        table = arrow_con.sql(f"SELECT a + a AS s FROM (SELECT '{value}'::DECIMAL(38,2) AS a)").to_arrow_table()
        assert table.schema.field("s").type == pa.decimal256(39, 2)
        assert table.column("s").to_pylist() == [exactly(lambda: value * 2)]

    def test_narrow_decimals_stay_decimal128(self, arrow_con):
        pa = pytest.importorskip("pyarrow")
        table = arrow_con.sql("SELECT 1.5::DECIMAL(38,2) AS v").to_arrow_table()
        assert table.schema.field("v").type == pa.decimal128(38, 2)

    @pytest.mark.parametrize(("width", "scale"), [(77, 3), (255, 20), (300, 37)])
    def test_beyond_decimal256_is_refused(self, arrow_con, width, scale):
        pytest.importorskip("pyarrow")
        rel = arrow_con.sql(f"SELECT '{decimal_of(width, scale)}'::DECIMAL({width},{scale}) AS v")
        with pytest.raises(duckdb.NotImplementedException, match="Arrow decimals support at most 76 digits"):
            rel.to_arrow_table()
        with pytest.raises(duckdb.NotImplementedException, match="Arrow decimals support at most 76 digits"):
            rel.to_arrow_reader()
        # the rows are still there in the forms that hold them
        assert rel.fetchall() == [(decimal_of(width, scale),)]

    def test_round_trip(self, arrow_con):
        pytest.importorskip("pyarrow")
        value = decimal_of(76, 10, negative=True)
        exported = arrow_con.sql(f"SELECT '{value}'::DECIMAL(76,10) AS v").to_arrow_table()  # noqa: F841
        rel = arrow_con.sql("SELECT v, typeof(v) FROM exported")
        assert rel.fetchall() == [(value, "DECIMAL(76,10)")]


class TestPolarsOutput:
    """Polars holds a decimal of at most 38 digits, so a wider one is refused by name."""

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(("width", "scale"), [(39, 5), (76, 10)])
    def test_wide_decimal_is_refused(self, arrow_con, lazy, width, scale):
        pytest.importorskip("polars")
        rel = arrow_con.sql(f"SELECT 1 AS i, '{decimal_of(width, scale)}'::DECIMAL({width},{scale}) AS v")
        with pytest.raises(
            duckdb.NotImplementedException,
            match=rf'column "v" is exported to Arrow as decimal256\({width}, {scale}\), which Polars does not read',
        ):
            rel.pl(lazy=lazy)

    def test_nested_wide_decimal_is_refused(self, arrow_con):
        pytest.importorskip("polars")
        rel = arrow_con.sql("SELECT [{'d': 1::DECIMAL(50,5)}] AS l")
        with pytest.raises(duckdb.NotImplementedException, match="a Polars decimal holds at most 38 digits"):
            rel.pl()

    @pytest.mark.parametrize("lazy", [False, True])
    def test_narrow_decimal(self, arrow_con, lazy):
        pl = pytest.importorskip("polars")
        frame = arrow_con.sql("SELECT 1.5::DECIMAL(38,2) AS v").pl(lazy=lazy)
        if lazy:
            frame = frame.collect()
        assert frame.schema["v"] == pl.Decimal(38, 2)
        assert frame["v"].to_list() == [Decimal("1.50")]

    def test_cast_to_a_narrower_decimal(self, arrow_con):
        pytest.importorskip("polars")
        frame = arrow_con.sql("SELECT (1.5::DECIMAL(50,2))::DECIMAL(38,2) AS v").pl()
        assert frame["v"].to_list() == [Decimal("1.50")]


class TestParameters:
    @pytest.mark.parametrize(("width", "scale"), SHAPES)
    @pytest.mark.parametrize("negative", [False, True])
    def test_round_trip(self, con, width, scale, negative):
        value = decimal_of(width, scale, negative=negative)
        assert con.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [(value, f"DECIMAL({width},{scale})")]
        assert con.execute("SELECT $v, typeof($v)", {"v": value}).fetchall() == [(value, f"DECIMAL({width},{scale})")]

    @pytest.mark.parametrize(
        ("value", "type_name"),
        [
            # 39 digits, where the 38 before were a DECIMAL and this was a DOUBLE
            (Decimal("1" + "0" * 38), "DECIMAL(39,0)"),
            (Decimal("-0." + "0" * 38 + "1"), "DECIMAL(40,39)"),
            # a positive exponent is that many zero digits
            (Decimal("123E+40"), "DECIMAL(43,0)"),
            (Decimal("-1E+38"), "DECIMAL(39,0)"),
            (Decimal("0E+50"), "DECIMAL(51,0)"),
            (Decimal("-0E-50"), "DECIMAL(51,50)"),
            (Decimal("1E-60"), "DECIMAL(61,60)"),
        ],
    )
    def test_shapes(self, con, value, type_name):
        result, result_type = con.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == type_name
        assert result == value
        assert isinstance(result, Decimal)

    @pytest.mark.parametrize("value", [Decimal("0E+50"), Decimal("-0E+50"), Decimal("-0E-50"), Decimal("0E-50")])
    def test_a_wide_zero_is_zero(self, con, value):
        assert con.execute("SELECT ? = 0, ? + 1, ? < 1, ? > -1", [value] * 4).fetchall() == [(True, 1, True, True)]
        distinct = "SELECT count(DISTINCT v) FROM (VALUES (?), (0::DECIMAL(51,0))) t(v)"
        assert con.execute(distinct, [value]).fetchall() == [(1,)]

    def test_narrow_values_are_unchanged(self, con):
        value = Decimal("9" * 38)
        assert con.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [(value, "DECIMAL(38,0)")]
        value = Decimal("-1.5")
        assert con.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [(value, "DECIMAL(2,1)")]

    def test_arithmetic_on_a_parameter(self, con):
        value = decimal_of(60, 5)
        assert con.execute("SELECT ? + 1, ? * 2, -?", [value, value, value]).fetchall() == [
            exactly(lambda: (value + 1, value * 2, -value))
        ]

    def test_comparison_with_a_parameter(self, con):
        value = decimal_of(60, 5)
        con.execute("CREATE TABLE t (v DECIMAL(60,5))")
        negative = decimal_of(60, 5, negative=True)
        con.execute("INSERT INTO t VALUES (?), (?), (NULL)", [value, negative])
        assert con.execute("SELECT v FROM t WHERE v = ?", [value]).fetchall() == [(value,)]
        assert con.execute("SELECT v FROM t WHERE v < ?", [value]).fetchall() == [(negative,)]

    def test_executemany(self, con):
        con.execute("CREATE TABLE t (i INTEGER, v DECIMAL(300,37))")
        rows = [
            (0, decimal_of(300, 37)),
            (1, decimal_of(300, 37, negative=True)),
            (2, None),
            (3, Decimal("1.5")),
            (4, decimal_of(39, 0)),
        ]
        con.executemany("INSERT INTO t VALUES (?, ?)", rows)
        assert con.execute("SELECT * FROM t ORDER BY i").fetchall() == rows

    def test_in_a_list_and_a_struct(self, con):
        wide = decimal_of(50, 5, negative=True)
        narrow = Decimal("1.5")
        row = con.execute("SELECT ?, typeof(?)", [[narrow, wide, None], [narrow, wide, None]]).fetchone()
        assert row == ([narrow, wide, None], "DECIMAL(50,5)[]")
        row = con.execute("SELECT ?, typeof(?)", [{"d": wide}, {"d": wide}]).fetchone()
        assert row == ({"d": wide}, "STRUCT(d DECIMAL(50,5))")

    def test_extreme_exponents_within_the_limit(self, con):
        large = Decimal("1E+20000")
        small = Decimal("-1E-20000")
        assert con.execute("SELECT typeof(?), typeof(?)", [large, small]).fetchall() == [
            ("DECIMAL(20001,0)", "DECIMAL(20001,20000)")
        ]
        assert con.execute("SELECT ?, ?", [large, small]).fetchall() == [(large, small)]

    @pytest.mark.parametrize(
        "value",
        [Decimal("1E+1000000"), Decimal("-1E-1000000"), Decimal("1E+999999999999999999"), Decimal("1E-99999999")],
    )
    def test_beyond_max_decimal_width(self, con, value):
        # max_decimal_width defaults to 1000000 digits; no DOUBLE stands in for a decimal beyond it
        with pytest.raises(
            duckdb.OutOfRangeException,
            match=r"the maximum DECIMAL width is 1000000 \(SET max_decimal_width to allow wider decimals\)",
        ):
            con.execute("SELECT ?", [value])

    def test_beyond_a_lowered_max_decimal_width(self, con):
        con.execute("SET max_decimal_width = 100")
        value = decimal_of(100, 0)
        assert con.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [(value, "DECIMAL(100,0)")]
        with pytest.raises(
            duckdb.OutOfRangeException,
            match=r"needs a DECIMAL of 101 digits, the maximum DECIMAL width is 100 \(SET max_decimal_width",
        ):
            con.execute("SELECT ?", [decimal_of(101, 0)])

    def test_raised_max_decimal_width(self, con):
        con.execute("SET max_decimal_width = 1100000")
        value = Decimal("1E-1000000")
        assert con.execute("SELECT typeof(?)", [value]).fetchall() == [("DECIMAL(1000001,1000000)",)]

    def test_max_decimal_width_38_keeps_the_double(self, con):
        # The setting of 38 is the mode that keeps the types of before wide decimals: there a decimal.Decimal of
        # more than 38 digits is still a DOUBLE
        con.execute("SET max_decimal_width = 38")
        value = decimal_of(39, 0)
        assert con.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [(float(value), "DOUBLE")]
        value = Decimal("9" * 38)
        assert con.execute("SELECT ?, typeof(?)", [value, value]).fetchall() == [(value, "DECIMAL(38,0)")]

    @pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
    def test_nan_and_infinity_are_float(self, con, value):
        # No DECIMAL holds these
        result, result_type = con.execute("SELECT ?, typeof(?)", [value, value]).fetchone()
        assert result_type == "FLOAT"
        assert math.isnan(result) if value.is_nan() else result == float(value)


class TestExpressions:
    @pytest.mark.parametrize(("width", "scale"), SHAPES)
    def test_constant_expression(self, con, width, scale):
        value = decimal_of(width, scale, negative=True)
        rel = con.sql("SELECT 1").select(duckdb.ConstantExpression(value).alias("v"))
        assert str(rel.types[0]) == f"DECIMAL({width},{scale})"
        assert rel.fetchall() == [(value,)]

    def test_filter_on_a_wide_constant(self, con):
        value = decimal_of(50, 5)
        rel = con.sql(f"SELECT v::DECIMAL(50,5) AS v FROM (VALUES ('{value}'), ('1.5')) t(v)")
        assert rel.filter(duckdb.ColumnExpression("v") == duckdb.ConstantExpression(value)).fetchall() == [(value,)]
        assert rel.filter(duckdb.ColumnExpression("v") < value).fetchall() == [(Decimal("1.50000"),)]

    def test_value(self, con):
        value = decimal_of(50, 5)
        typed = duckdb.Value(value, duckdb.decimal_type(50, 5))
        assert con.execute("SELECT ?, typeof(?)", [typed, typed]).fetchall() == [(value, "DECIMAL(50,5)")]

    def test_values_relation(self, con):
        wide = decimal_of(50, 5)
        rel = con.values([Decimal("1.5"), wide])
        assert [str(t) for t in rel.types] == ["DECIMAL(2,1)", "DECIMAL(50,5)"]
        assert rel.fetchall() == [(Decimal("1.5"), wide)]

    def test_beyond_max_decimal_width(self, con):
        with pytest.raises(duckdb.Error, match="max_decimal_width"):
            con.sql("SELECT 1").select(duckdb.ConstantExpression(Decimal("1E+1000000"))).fetchall()


class TestPandasObjectColumn:
    @pytest.mark.parametrize(("width", "scale"), SHAPES)
    def test_round_trip(self, con, width, scale):
        pd = pytest.importorskip("pandas")
        positive = decimal_of(width, scale)
        negative = decimal_of(width, scale, negative=True)
        frame = pd.DataFrame({"v": [positive, None, negative]})  # noqa: F841
        rel = con.sql("SELECT v FROM frame")
        assert str(rel.types[0]) == f"DECIMAL({width},{scale})"
        assert rel.fetchall() == [(positive,), (None,), (negative,)]
        # and back out, as the float64 a DECIMAL column is in pandas
        out = rel.df()
        assert str(out["v"].dtype) == "float64"
        assert out["v"][0] == float(positive)
        assert math.isnan(out["v"][1])

    def test_mixed_widths_combine(self, con):
        pd = pytest.importorskip("pandas")
        values = [Decimal("1.5"), decimal_of(50, 0, negative=True), Decimal("0." + "0" * 44 + "1"), None]
        frame = pd.DataFrame({"v": values})  # noqa: F841
        rel = con.sql("SELECT v FROM frame")
        # 50 digits before the point and 45 after it
        assert str(rel.types[0]) == "DECIMAL(95,45)"
        assert [row[0] for row in rel.fetchall()] == values

    def test_in_a_list(self, con):
        pd = pytest.importorskip("pandas")
        wide = decimal_of(50, 5)
        frame = pd.DataFrame({"v": [[Decimal("1.5"), wide], [None]]})  # noqa: F841
        rel = con.sql("SELECT v FROM frame")
        assert str(rel.types[0]) == "DECIMAL(50,5)[]"
        assert rel.fetchall() == [([Decimal("1.5"), wide],), ([None],)]

    def test_insert_into_a_table(self, con):
        pd = pytest.importorskip("pandas")
        values = [decimal_of(60, 5), decimal_of(60, 5, negative=True), None]
        frame = pd.DataFrame({"v": values})  # noqa: F841
        con.execute("CREATE TABLE t (v DECIMAL(60,5))")
        con.execute("INSERT INTO t SELECT v FROM frame")
        assert [row[0] for row in con.execute("SELECT v FROM t").fetchall()] == values

    def test_beyond_max_decimal_width(self, con):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"v": [Decimal("1E+1000000")]})  # noqa: F841
        with pytest.raises(duckdb.OutOfRangeException, match="the maximum DECIMAL width is 1000000"):
            con.sql("SELECT v FROM frame").fetchall()

    def test_max_decimal_width_38_keeps_the_double(self, con):
        pd = pytest.importorskip("pandas")
        con.execute("SET max_decimal_width = 38")
        frame = pd.DataFrame({"v": [Decimal("1.5"), decimal_of(39, 0)]})  # noqa: F841
        rel = con.sql("SELECT v FROM frame")
        assert str(rel.types[0]) == "DOUBLE"
        assert rel.fetchall() == [(1.5,), (float(decimal_of(39, 0)),)]


class TestAppender:
    def test_append_rows(self, con):
        con.execute("CREATE TABLE t (k INTEGER, v DECIMAL(60,5))")
        wide = decimal_of(60, 5, negative=True)
        with con.appender("t") as appender:
            appender.append_row(1, wide)
            appender.append_row(2, None)
            appender.append_row(3, Decimal("1.5"))
            appender.append_row(4, 7)
        assert con.execute("SELECT * FROM t ORDER BY k").fetchall() == [
            (1, wide),
            (2, None),
            (3, Decimal("1.50000")),
            (4, Decimal("7.00000")),
        ]

    def test_a_value_beyond_the_column_is_refused(self, con):
        con.execute("CREATE TABLE t (v DECIMAL(60,5))")
        with pytest.raises(duckdb.ConversionException, match=r"DECIMAL\(60,5\)"), con.appender("t") as appender:
            appender.append_row(decimal_of(56, 0))

    def test_query_appender(self, con):
        con.execute("CREATE TABLE t (k INTEGER, v DECIMAL(60,5))")
        wide = decimal_of(60, 5)
        with con.query_appender(
            "INSERT INTO t FROM appender_data", {"k": "INTEGER", "v": "DECIMAL(60, 5)"}
        ) as appender:
            appender.append_row(1, wide)
        assert con.execute("SELECT * FROM t").fetchall() == [(1, wide)]


class TestStorage:
    def test_table_round_trip_on_disk(self, tmp_path):
        value = decimal_of(255, 20, negative=True)
        path = tmp_path / "wide.duckdb"
        # A wide DECIMAL column needs the storage version it was added in
        with duckdb.connect(path, config={"storage_compatibility_version": "v2.1.0"}) as con:
            con.execute("CREATE TABLE t (v DECIMAL(255,20))")
            con.execute("INSERT INTO t VALUES (?), (NULL)", [value])
        with duckdb.connect(path) as con:
            assert con.execute("SELECT v FROM t").fetchall() == [(value,), (None,)]
            assert con.execute("SELECT count(*) FROM t WHERE v = ?", [value]).fetchall() == [(1,)]


class TestUDF:
    def test_native(self, con):
        kind = duckdb.decimal_type(50, 5)
        seen = []

        def negate(value):
            seen.append(value)
            return exactly(lambda: -value)

        con.create_function("negate", negate, [kind], kind)
        value = decimal_of(50, 5)
        rel = con.sql(f"SELECT negate(v::DECIMAL(50,5)) AS r FROM (VALUES ('{value}'), ('-1.5')) t(v)")
        assert str(rel.types[0]) == "DECIMAL(50,5)"
        assert rel.fetchall() == [(exactly(lambda: -value),), (Decimal("1.50000"),)]
        assert sorted(seen) == [Decimal("-1.50000"), value]

    def test_native_returns_a_wider_value_than_declared(self, con):
        con.create_function("too_wide", lambda: decimal_of(60, 0), [], duckdb.decimal_type(50, 0))
        with pytest.raises(duckdb.Error, match=r"DECIMAL\(50,0\)"):
            con.sql("SELECT too_wide()").fetchall()

    def test_native_narrow_value_for_a_wide_type(self, con):
        con.create_function("small", lambda: Decimal("1.5"), [], duckdb.decimal_type(300, 37))
        assert con.sql("SELECT small()").fetchall() == [(Decimal("1." + "5".ljust(37, "0")),)]

    def test_arrow(self, arrow_con):
        pa = pytest.importorskip("pyarrow")
        pc = pytest.importorskip("pyarrow.compute")
        kind = duckdb.decimal_type(60, 5)
        seen = []

        def negate(column):
            seen.append(column.type)
            return pc.negate(column)

        arrow_con.create_function("negate", negate, [kind], kind, type="arrow")
        value = decimal_of(60, 5)
        rel = arrow_con.sql(f"SELECT negate(v::DECIMAL(60,5)) AS r FROM (VALUES ('{value}'), (NULL)) t(v)")
        assert rel.fetchall() == [(exactly(lambda: -value),), (None,)]
        assert seen[0] == pa.decimal256(60, 5)
