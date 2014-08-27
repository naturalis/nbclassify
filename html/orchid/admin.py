from django.contrib import admin

from orchid.models import Photo, Identity

class IdentityInline(admin.TabularInline):
    model = Identity
    extra = 0

class PhotoAdmin(admin.ModelAdmin):
    list_display = ('file_name','id')
    readonly_fields = ('image_tag',)
    inlines = [IdentityInline]

admin.site.register(Photo, PhotoAdmin)
