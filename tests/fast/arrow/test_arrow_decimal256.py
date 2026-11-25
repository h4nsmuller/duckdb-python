from decimal import Decimal

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")


class TestArrowDecimal256:
    def test_decimal_256_of_a_narrow_precision(self, duckdb_cursor):
        with duckdb.connect() as conn:
            pa_decimal256 = pa.Table.from_pylist(  # noqa: F841
                [{"data": Decimal("100.00")} for _ in range(4)],
                pa.schema([("data", pa.decimal256(12, 4))]),
            )
            rel = conn.sql("select * from pa_decimal256")
            assert [str(t) for t in rel.types] == ["DECIMAL(12,4)"]
            assert rel.fetchall() == [(Decimal("100.0000"),)] * 4

    @pytest.mark.parametrize(("precision", "scale"), [(39, 0), (50, 5), (76, 0), (76, 38), (76, 76)])
    def test_decimal_256_is_exact(self, precision, scale):
        digits = "".join(str(i % 9 + 1) for i in range(precision))
        text = (digits[: precision - scale] or "0") + ("." + digits[precision - scale :] if scale else "")
        values = [Decimal(text), None, Decimal("-" + text), Decimal(0).scaleb(-scale)]
        with duckdb.connect() as conn:
            pa_decimal256 = pa.table({"data": pa.array(values, pa.decimal256(precision, scale))})  # noqa: F841
            rel = conn.sql("select * from pa_decimal256")
            assert [str(t) for t in rel.types] == [f"DECIMAL({precision},{scale})"]
            assert rel.fetchall() == [(value,) for value in values]
            # and back out as a decimal256, which is in the Arrow format from version 1.1 on
            conn.execute("SET arrow_output_version = '1.1'")
            assert rel.to_arrow_table().column("data").to_pylist() == values
            if precision > 38:
                assert rel.to_arrow_table().schema.field("data").type == pa.decimal256(precision, scale)

    def test_decimal_256_filter_is_pushed_down(self):
        values = [Decimal("1.5"), None, Decimal("-" + "9" * 70 + ".123456"), Decimal("9" * 70 + ".000001")]
        ds = pytest.importorskip("pyarrow.dataset")
        with duckdb.connect() as conn:
            pa_decimal256 = ds.dataset(pa.table({"data": pa.array(values, pa.decimal256(76, 6))}))  # noqa: F841
            rel = conn.sql("select * from pa_decimal256 where data > 1.5")
            assert rel.fetchall() == [(values[3],)]
            # the dataset applies the filter, so none is left above the scan
            plan = conn.sql("explain select * from pa_decimal256 where data > 1.5").fetchall()[0][1]
            assert "Filters" in plan
            assert "Expression" not in plan
            limit = Decimal("-" + "9" * 70 + ".123456")
            assert conn.execute("select * from pa_decimal256 where data <= ?", [limit]).fetchall() == [(limit,)]
