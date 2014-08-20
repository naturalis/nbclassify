from django.contrib import admin

from orchid.models import Photo

class PhotoAdmin(admin.ModelAdmin):
    list_display= ('file_name',)
    readonly_fields = ('image_tag',)

admin.site.register(Photo, PhotoAdmin)
