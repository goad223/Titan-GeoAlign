"""Bilingual (English/Arabic) exception hierarchy for ChangeMaster Ultimate.

Every exception carries a message in both English and Arabic so the GUI and
logs can present errors in the user's preferred language. The ``str()`` form
always contains both languages, separated by `` | ``.
"""

from __future__ import annotations


class ChangeMasterError(Exception):
    """Base class for all ChangeMaster errors with bilingual messages.

    Attributes:
        message_en: Human-readable English message.
        message_ar: Human-readable Arabic message.
    """

    default_en: str = "An unexpected error occurred."
    default_ar: str = "حدث خطأ غير متوقع."

    def __init__(self, message_en: str | None = None, message_ar: str | None = None) -> None:
        """Initialize the error with optional English and Arabic messages.

        Args:
            message_en: English message; falls back to the class default.
            message_ar: Arabic message; falls back to the class default.
        """
        self.message_en: str = message_en or self.default_en
        self.message_ar: str = message_ar or self.default_ar
        super().__init__(f"{self.message_en} | {self.message_ar}")

    def bilingual(self) -> dict[str, str]:
        """Return both messages keyed by language code (``en``/``ar``)."""
        return {"en": self.message_en, "ar": self.message_ar}


class HardwareDetectionError(ChangeMasterError):
    """Raised when hardware probing fails irrecoverably."""

    default_en = "Failed to detect system hardware."
    default_ar = "فشل في كشف عتاد النظام."


class ConfigError(ChangeMasterError):
    """Raised for configuration loading/saving problems."""

    default_en = "Configuration error."
    default_ar = "خطأ في الإعدادات."


class MissingDependencyError(ChangeMasterError):
    """Raised when an optional heavy dependency is required but not installed."""

    default_en = "A required optional dependency is not installed."
    default_ar = "اعتمادية اختيارية مطلوبة غير مثبتة."

    def __init__(self, package: str, feature: str | None = None) -> None:
        """Build a helpful bilingual message naming the missing package.

        Args:
            package: pip package name that is missing (e.g. ``rasterio``).
            feature: Optional short description of the blocked feature.
        """
        self.package: str = package
        self.feature: str | None = feature
        feat_en = f" (required for: {feature})" if feature else ""
        feat_ar = f" (مطلوبة لـ: {feature})" if feature else ""
        super().__init__(
            f"Optional dependency '{package}' is not installed{feat_en}. "
            f"Install it with: pip install {package}",
            f"الاعتمادية الاختيارية '{package}' غير مثبتة{feat_ar}. "
            f"ثبّتها بالأمر: pip install {package}",
        )


class ReaderError(ChangeMasterError):
    """Base class for image reading failures."""

    default_en = "Failed to read the image file."
    default_ar = "فشل في قراءة ملف الصورة."


class UnsupportedFormatError(ReaderError):
    """Raised when no registered reader supports a given file."""

    default_en = "The file format is not supported."
    default_ar = "صيغة الملف غير مدعومة."

    def __init__(self, path: str) -> None:
        """Create the error for an unsupported file path."""
        self.path: str = path
        super().__init__(
            f"No reader available for file: {path}",
            f"لا يوجد قارئ متاح للملف: {path}",
        )


class FileAccessError(ReaderError):
    """Raised when a file does not exist or cannot be opened."""

    default_en = "The file does not exist or cannot be accessed."
    default_ar = "الملف غير موجود أو لا يمكن الوصول إليه."

    def __init__(self, path: str) -> None:
        """Create the error for an inaccessible file path."""
        self.path: str = path
        super().__init__(
            f"File not found or inaccessible: {path}",
            f"الملف غير موجود أو لا يمكن الوصول إليه: {path}",
        )


class MetadataError(ReaderError):
    """Raised when product metadata (manifest/MTL) is missing or malformed."""

    default_en = "Product metadata is missing or malformed."
    default_ar = "بيانات المنتج الوصفية مفقودة أو تالفة."


class BandError(ReaderError):
    """Raised when a requested band index/name is invalid."""

    default_en = "Requested band is invalid."
    default_ar = "النطاق المطلوب غير صالح."

    def __init__(self, band: int | str, available: int | list[str]) -> None:
        """Create the error describing the invalid band request.

        Args:
            band: Band index (1-based) or name that was requested.
            available: Number of bands or list of valid band names.
        """
        self.band = band
        super().__init__(
            f"Invalid band '{band}'. Available: {available}",
            f"النطاق '{band}' غير صالح. المتاح: {available}",
        )


class WriterError(ChangeMasterError):
    """Raised when writing an output file fails."""

    default_en = "Failed to write the output file."
    default_ar = "فشل في كتابة ملف الإخراج."


class TileAccessError(ChangeMasterError):
    """Raised for invalid tiled-access parameters (e.g. bad tile size)."""

    default_en = "Invalid tiled access parameters."
    default_ar = "معاملات القراءة المجزأة غير صالحة."


class SensorProfileError(ChangeMasterError):
    """Raised when a sensor profile is missing or malformed."""

    default_en = "Sensor profile error."
    default_ar = "خطأ في بروفايل المستشعر."
